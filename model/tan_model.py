import torch
from torch import nn
import torch.nn.functional as F 
from torch.nn.utils.rnn import pad_sequence
from torch.nn import LayerNorm
from collections import OrderedDict
from transformers import BertModel, DistilBertModel
import numpy as np
from tfm_model import TemporalEncoder, get_position_embedding_sine
from word2vec_model import Word2VecModel


class TemporalAligner(nn.Module):
    def __init__(self, 
                 num_encoder_layers=2, 
                 num_decoder_layers=2, 
                 sim='cos', 
                 language_model='word2vec',
                 pos_enc='learned',
                 use_text_pos_enc=0,
                 return_dual_feature=1,
                 random_pos_start=1,
                 use_alignability_head=0,
                 ):
        super().__init__()
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.sim = sim 
        self.pos_enc = pos_enc
        self.language_model = language_model
        self.use_text_pos_enc = use_text_pos_enc
        print(f'Use textual pos-enc in joint-encoder = {bool(use_text_pos_enc)}')
        self.return_dual_feature = return_dual_feature
        self.random_pos_start = random_pos_start
        self.use_alignability_head = use_alignability_head

        if language_model == 'bert':
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        elif language_model == 'word2vec':
            self.bert = Word2VecModel()
        text_embed_dim = {'bert': 768, 'word2vec': 512}

        self.video_temporal_encoder = TemporalEncoder(
            width=512, layers=num_encoder_layers, heads=8)
        self.joint_temporal_encoder = TemporalEncoder(
            width=512, layers=num_decoder_layers, heads=8)

        self.video_pre_proj = nn.Linear(1024, 512, bias=False)
        self.text_pre_proj = nn.Linear(text_embed_dim[language_model], 512, bias=False)
        self.ln_text_init = LayerNorm(512)
        self.ln_video_init = LayerNorm(512)
        self.ln_position_init = LayerNorm(512)
        self.ln_video_post_enc = LayerNorm(512)
        self.ln_joint_post_enc = LayerNorm(512)
        
        # temporal positional encoding for video
        if self.pos_enc == 'learned':
            self.temporal_pos_embed = nn.Parameter(torch.empty(1024, 512))
            nn.init.normal_(self.temporal_pos_embed, std=0.01)
        elif self.pos_enc == 'sine':
            temporal_pos_embed = get_position_embedding_sine(512, 1024)
            self.register_buffer('temporal_pos_embed', temporal_pos_embed)

        # temporal positional encoding for text
        self.text_temporal_pos_embed = nn.Parameter(torch.empty(1024, 512))
        nn.init.normal_(self.text_temporal_pos_embed, std=0.01)

        self.mlp = nn.Linear(512, 512)
        if self.use_alignability_head:
            self.binary_head = nn.Linear(512, 1)
            nn.init.normal_(self.binary_head.weight, std=0.01)
            nn.init.zeros_(self.binary_head.bias)
        self.initialize_parameters()


    def initialize_parameters(self):
        nn.init.normal_(self.video_pre_proj.weight, std=0.01)
        nn.init.normal_(self.text_pre_proj.weight, std=0.01)
        for name, param in self.mlp.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            elif 'weight' in name:
                nn.init.normal_(param, std=0.01)

        proj_std = (self.joint_temporal_encoder.width ** -0.5) * ((2 * self.joint_temporal_encoder.layers) ** -0.5)
        attn_std = self.joint_temporal_encoder.width ** -0.5
        fc_std = (2 * self.joint_temporal_encoder.width) ** -0.5
        for block in self.video_temporal_encoder.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        for block in self.joint_temporal_encoder.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)


    def forward(self, video_embed, lang_embed, 
                video_padding_mask, lang_padding_mask,
                text_timestamp,
                interpolate_from=None):
        ### Dual Encoder ###
        # video attend itself with pos-enc
        B,T,_ = video_embed.shape
        video_out = self.get_visual_feature(
            video_embed, 
            video_padding_mask, 
            interpolate_from)

        # text embedding without temporal-enc
        lang_embed_raw = self.get_textual_feature(lang_embed)

        # get cosine distance for Dual Encoder
        video_feature_norm = video_out / video_out.norm(dim=-1, keepdim=True)
        text_feature_norm = lang_embed_raw / lang_embed_raw.norm(dim=-1, keepdim=True)
        contrastive_logits_dual = torch.einsum("astc,bkc->astbk", 
            video_feature_norm, text_feature_norm)

        ### Joint Encoder ###
        # get text embedding with/without temporal pos-enc
        if self.use_text_pos_enc:
            lang_embed_with_time = self.get_textual_feature_with_time(lang_embed, 
                text_timestamp, interpolate_from)
        else:
            lang_embed_with_time = lang_embed_raw

        # get video and text embedding after joint model    
        joint_video_out, joint_text_out = self.get_joint_feature(
            video_embed, video_padding_mask,
            lang_embed_with_time, lang_padding_mask,
            interpolate_from)
        
        # get cosine distance for Joint Encoder
        video_feature_norm_joint = joint_video_out / joint_video_out.norm(dim=-1, keepdim=True)
        text_feature_norm_joint = joint_text_out / joint_text_out.norm(dim=-1, keepdim=True)
        contrastive_logits_joint = torch.einsum("astc,bskc->astbk", 
            video_feature_norm_joint, text_feature_norm_joint)

        output_dict = {'logits_dual': contrastive_logits_dual, 'logits_joint': contrastive_logits_joint}
        
        if self.return_dual_feature:
            output_dict['dual_feature_video'] = video_feature_norm
            output_dict['dual_feature_text'] = text_feature_norm
        if self.use_alignability_head:
            output_dict['dual_logits_alignability'] = self.binary_head(lang_embed_raw)
            output_dict['joint_logits_alignability'] = self.binary_head(joint_text_out)
        return output_dict


    def get_visual_feature(self, video_embed, video_padding_mask, interpolate_from=None):
        """Get video embedding from video transformer encoder in the dual model.
        No text inputs. Can be used for retrieval setting"""
        video_embed = self.ln_video_init(self.video_pre_proj(video_embed))
        B,T,_ = video_embed.shape
        if interpolate_from:
            video_pos_embed_source = self.temporal_pos_embed[None, 0:interpolate_from, :]
            video_pos_embed = F.interpolate(video_pos_embed_source.transpose(1,2), 
                size=T, mode='linear', align_corners=False).transpose(1,2)
        else:
            if self.random_pos_start:
                pos_start_idx = np.random.randint(0, int(T/2))
            else:
                pos_start_idx = 0
            video_pos_embed = self.temporal_pos_embed[None, pos_start_idx:pos_start_idx+T, :]
        video_embed = video_embed + self.ln_position_init(video_pos_embed)
        video_embed = video_embed.permute(1,0,2) # BTC -> TBC

        if self.num_encoder_layers > 0:
            video_encoder_out = self.video_temporal_encoder(
                video_embed, video_padding_mask
            )
            video_encoder_out[-1] = self.ln_video_post_enc(video_encoder_out[-1])
            video_out = video_encoder_out[-1]
            return torch.stack(video_encoder_out, dim=1).permute(2,1,0,3)  # B,Stage,T,C
        else:
            video_out = video_embed
            return video_out


    def get_joint_feature(self, video_embed, video_padding_mask,
                          lang_embed_with_time, lang_padding_mask,
                          interpolate_from=None):
        """Get the joint video embedding and text embedding from the joint encoder.
        It takes both visual and textual inputs."""
        video_embed = self.ln_video_init(self.video_pre_proj(video_embed))
        B,T,_ = video_embed.shape
        if interpolate_from:
            video_pos_embed_source = self.temporal_pos_embed[None, 0:interpolate_from, :]
            video_pos_embed = F.interpolate(video_pos_embed_source.transpose(1,2), 
                size=T, mode='linear', align_corners=False).transpose(1,2)
        else:
            if self.random_pos_start:
                pos_start_idx = np.random.randint(0, int(T/2))
            else:
                pos_start_idx = 0
            video_pos_embed = self.temporal_pos_embed[None, pos_start_idx:pos_start_idx+T, :]
        video_embed_with_time = video_embed + self.ln_position_init(video_pos_embed)

        joint_embed = torch.cat((video_embed_with_time, lang_embed_with_time), dim=1)
        joint_embed = joint_embed.permute(1,0,2) # BXC -> XBC
        joint_padding_mask = torch.cat((video_padding_mask, lang_padding_mask), dim=1)
        
        joint_output = self.joint_temporal_encoder(joint_embed, joint_padding_mask)
        joint_output[-1] = self.ln_joint_post_enc(joint_output[-1])

        joint_output = torch.stack(joint_output, dim=1).permute(2,1,0,3)  # B,Stage,X,C
        return joint_output[:,:,0:T], joint_output[:,:,T::], 


    def get_textual_feature_with_time(self, lang_embed, text_timestamp, interpolate_from=None):
        """add proper positional embedding to text
        lang_embed: tensor [B,N,C]
        text_timestamp: B,N,T, binary """
        text_proj = self.ln_text_init(self.text_pre_proj(lang_embed))
        N = lang_embed.shape[1]
        if interpolate_from:
            text_pos_embed_source = self.text_temporal_pos_embed[None, 0:interpolate_from, :]
            text_pos_embed = F.interpolate(text_pos_embed_source.transpose(1,2), 
                size=N, mode='linear', align_corners=False).transpose(1,2)
        else:
            if self.random_pos_start:
                pos_start_idx = np.random.randint(0, int(N/2))
            else:
                pos_start_idx = 0
            text_pos_embed = self.text_temporal_pos_embed[None, pos_start_idx:pos_start_idx+N, :]
        return text_proj + self.ln_position_init(text_pos_embed)


    def get_textual_feature(self, lang_embed):
        """get text embedding after proj and LayerNorm"""
        text_proj = self.ln_text_init(self.text_pre_proj(lang_embed))
        return text_proj

    
    def get_text_visual_sim_joint(self, video_embed, lang_embed, interpolate_from=None):
        if isinstance(interpolate_from, list) or isinstance(interpolate_from, tuple):
            assert len(interpolate_from) == 2
            text_interpolate_from = interpolate_from[1]
            interpolate_from = interpolate_from[0]
        else:
            text_interpolate_from = None

        if self.use_text_pos_enc:
            lang_embed_with_time = self.get_textual_feature_with_time(lang_embed, 
                text_timestamp=None,
                interpolate_from=text_interpolate_from)
        else:
            lang_embed_with_time = self.get_textual_feature(lang_embed)
        B,T,_ = video_embed.shape
        N = lang_embed_with_time.shape[1]
        lang_padding_mask = torch.zeros(B,N,device=video_embed.device).bool()
        video_padding_mask = torch.zeros(B,T,device=video_embed.device).bool()
        joint_video_out, joint_text_out = self.get_joint_feature(
            video_embed, video_padding_mask,
            lang_embed_with_time, lang_padding_mask,
            interpolate_from)

        video_feature_norm_joint = joint_video_out / joint_video_out.norm(dim=-1, keepdim=True)
        text_feature_norm_joint = joint_text_out / joint_text_out.norm(dim=-1, keepdim=True)
        contrastive_logits_joint = torch.einsum("bstc,bskc->bstk", 
            video_feature_norm_joint, text_feature_norm_joint)
        return contrastive_logits_joint

    
    def get_text_visual_sim_dual(self, video_embed, lang_embed, interpolate_from=None):
        lang_embed_raw = self.get_textual_feature(lang_embed)
        B,T,_ = video_embed.shape
        N = lang_embed_raw.shape[1]
        video_padding_mask = torch.zeros(B,T,device=video_embed.device).bool()

        video_out = self.get_visual_feature(
            video_embed, 
            video_padding_mask, 
            interpolate_from)

        video_feature_norm = video_out / video_out.norm(dim=-1, keepdim=True)
        text_feature_norm = lang_embed_raw / lang_embed_raw.norm(dim=-1, keepdim=True)
        contrastive_logits_dual = torch.einsum("bstc,bkc->bstk", 
            video_feature_norm, text_feature_norm)

        return contrastive_logits_dual
        

    def get_alignability(self, video_embed, lang_embed, interpolate_from=None):
        if isinstance(interpolate_from, list) or isinstance(interpolate_from, tuple):
            assert len(interpolate_from) == 2
            text_interpolate_from = interpolate_from[1]
            interpolate_from = interpolate_from[0]
        else:
            text_interpolate_from = None

        if self.use_text_pos_enc:
            lang_embed_with_time = self.get_textual_feature_with_time(lang_embed, 
                text_timestamp=None,
                interpolate_from=text_interpolate_from)
        else:
            lang_embed_with_time = self.get_textual_feature(lang_embed)
        B,T,_ = video_embed.shape
        N = lang_embed_with_time.shape[1]
        lang_padding_mask = torch.zeros(B,N,device=video_embed.device).bool()
        video_padding_mask = torch.zeros(B,T,device=video_embed.device).bool()
        _, joint_text_out = self.get_joint_feature(
            video_embed, video_padding_mask,
            lang_embed_with_time, lang_padding_mask,
            interpolate_from)

        dual_alignability = self.binary_head(self.get_textual_feature(lang_embed))
        joint_alignability = self.binary_head(joint_text_out)
        return {'alignability-dual': dual_alignability,
                'alignability-joint': joint_alignability}


class TwinTemporalAligner(nn.Module):
    """Duplicate TemporalAligner for EMA."""
    def __init__(self, m=0.999, *args, **kwargs):
        super().__init__()
        self.m = m
        self.online = TemporalAligner(*args, **kwargs)  # update by backprop
        self.target = TemporalAligner(*args, **kwargs)  # update by EMA
        self._copy_param()
        self.bert = self.online.bert
        self.get_visual_feature = self.online.get_visual_feature
        self.get_joint_feature = self.online.get_joint_feature
        self.get_textual_feature_with_time = self.online.get_textual_feature_with_time
        self.get_textual_feature = self.online.get_textual_feature
        self.get_text_visual_sim = self.online.get_text_visual_sim
        self.get_text_visual_sim_dual = self.online.get_text_visual_sim_dual
        self.get_alignability = self.online.get_alignability 

        # turn off online branch's random pos enc
        self.target.random_pos_start = 0

    def _copy_param(self):
        for param_online, param_target in zip(self.online.parameters(), self.target.parameters()):
            param_target.data.copy_(param_online.data)  # initialize
            param_target.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        '''Momentum update of the target encoder'''
        for param_online, param_target in zip(self.online.parameters(), self.target.parameters()):
            param_target.data = param_target.data * self.m + param_online.data * (1. - self.m)
    
    def forward(self, *args, **kwargs):
        return self.online(*args, **kwargs)

    @torch.no_grad()
    def forward_from_ema(self, *args, **kwargs):
        return self.target(*args, **kwargs)

