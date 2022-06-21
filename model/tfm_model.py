"""Transformer building blocks.
Code is modified from https://github.com/openai/CLIP/blob/main/clip/model.py """

import math
import torch
from torch import nn
from torch.nn import LayerNorm
from collections import OrderedDict


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


### Encoder ###
class ResidualAttentionBlock_Step(nn.Module):
    def __init__(self, d_model: int, n_head: int,):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        key_padding_mask = key_padding_mask.to(device=x.device) if key_padding_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=key_padding_mask)[0]

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        x_norm = self.ln_1(x)
        x = x + self.attention(x_norm, key_padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return x, x_norm


class TemporalEncoder(nn.Module):
    def __init__(self, width: int, layers: int, heads: int,):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock_Step(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        intermediate = []
        for block in self.resblocks:
            x, x_norm = block(x, key_padding_mask)
            intermediate.append(x_norm)
        intermediate.pop(0)
        intermediate.append(x)
        return intermediate


### Decoder ###
class ResidualDecoderBlock_Step(nn.Module):
    def __init__(self, d_model, n_head,):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.ln_3 = LayerNorm(d_model)

    def self_attention(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        key_padding_mask = key_padding_mask.to(device=x.device) if key_padding_mask is not None else None
        return self.self_attn(x, x, x, need_weights=False, key_padding_mask=key_padding_mask)[0]

    def attention(self, x: torch.Tensor, memory: torch.Tensor, key_padding_mask: torch.Tensor = None):
        key_padding_mask = key_padding_mask.to(device=x.device) if key_padding_mask is not None else None
        return self.attn(x, memory, memory, need_weights=False, key_padding_mask=key_padding_mask)[0]

    def forward(self, x, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x_norm = self.ln_1(x)
        x = x + self.self_attention(x_norm, key_padding_mask=tgt_key_padding_mask)
        x = x + self.attention(self.ln_2(x), memory, key_padding_mask=memory_key_padding_mask)
        x = x + self.mlp(self.ln_3(x))
        return x, x_norm


class TemporalDecoder(nn.Module):
    def __init__(self, width: int, layers: int, heads: int,):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualDecoderBlock_Step(width, heads) for _ in range(layers)])

    def forward(self, x, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        intermediate = []
        for block in self.resblocks:
            x, x_norm = block(x, memory, tgt_key_padding_mask, memory_key_padding_mask)
            intermediate.append(x_norm)
        intermediate.pop(0)
        intermediate.append(x)
        return intermediate


class PositionEmbeddingSine(nn.Module):
    """ This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, 1D version """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):
        assert mask is not None # B,T
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_y = y_embed[:, :, None] / dim_t
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = pos_y.permute(0,2,1)
        return pos


def get_position_embedding_sine(feature_dim=512, num_features=1024, temperature=10000):
    scale = 2 * math.pi
    embed = torch.arange(num_features)
    eps = 1e-6
    embed = embed / (embed[-1:] + eps) * scale
    dim_t = torch.arange(feature_dim, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / feature_dim)

    embed = embed[:,None] / dim_t
    embed = torch.stack((embed[:,0::2].sin(), embed[:,1::2].cos()), dim=2).flatten(1)
    embed = embed.permute(0,1)
    return embed  # num_features, feature_dim

