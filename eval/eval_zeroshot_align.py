import os
import sys
sys.path.append('../')
sys.path.append('../../')

import numpy as np
from tqdm import tqdm 
import json 
import matplotlib.pyplot as plt 
import pandas as pd 
import math 
import torch

from utils.data_utils import DataLoaderFast, DataLoaderBG
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate
from sklearn import metrics 


def read_file(path):
    with open(path, 'r') as f:
        content = f.readlines()
    content = [i.strip() for i in content]
    return content

def read_json(path):
    with open(path, 'r') as f:
        content = json.load(f)
    return content


class HTM_Align():
    """HTM_Align dataset. 
    For each video, return all the visual features and all the texts."""
    def __init__(self,
                 source='htm_align.json',
                 video_feature_path = '/scratch/shared/beegfs/htd/DATA/HowTo100M/howto100m_s3d_features',
                 num_clips=4,
                 seq_len=64,
                 ds=1):
        self.num_clips = num_clips
        self.seq_len = seq_len
        self.ds = ds
        self.video_feature_path = video_feature_path

        anno_path = f'{os.path.dirname(os.path.abspath(__file__))}/../data/{source}'
        with open(anno_path) as fp: anno = json.load(fp)
        self.anno = anno 
        
        for i in self.anno.keys():
            assert os.path.exists(os.path.join(self.video_feature_path, "{}.mp4.npy".format(i)))
        self.video_info = sorted(self.anno.keys())

    def __len__(self):
        return len(self.video_info)

    def __getitem__(self, idx):
        vid = self.video_info[idx]
        anno = self.anno[vid]
        text, text_start, text_end, text_aligned = [],[],[],[]
        for seg in anno:
            text_aligned.append(seg[0])
            text_start.append(seg[1])
            text_end.append(seg[2])
            text.append(seg[3])

        video = self._get_video_feature(vid, text_start, text_end, self.num_clips)
        return {'video': video,
                'start': torch.tensor(text_start),
                'end': torch.tensor(text_end),
                'vid':vid,
                'str':text,
                'aligned': torch.tensor(text_aligned)}

    def _get_video_feature(self, vid, start, end, num_clips=4):
        path = os.path.join(self.video_feature_path, "{}.mp4.npy".format(vid))
        feature = torch.from_numpy(np.load(path))
        vlen = feature.size(0) 

        if self.seq_len == -1: # take full length
            return feature.float()
        else:
            raise NotImplementedError


@torch.no_grad()
def test_alignment_htm(get_text_visual_sim, device, args):
    D = HTM_Align(seq_len=-1, source='htm_align.json')
    data_loader = DataLoaderFast(D, batch_size=1, num_workers=0)

    recall = []
    total_vlen = []
    total_text_count = []
    total_aligned_count = []

    total_align_sim = []
    total_align_tgt = []

    seq_len = args.seq_len
    method = 'overlap-seq'  # 'overlap-seq' or 'interpolate'
    print(f'Test Alignment with {method} method')

    for input_data in tqdm(data_loader, total=len(data_loader)):
        video = input_data['video'].to(device)
        text_str = [i[0] for i in input_data['str']]
        tgt_aligned = input_data["aligned"][0].tolist()
        vid = input_data['vid'][0]

        text_str_aligned = np.array(text_str)[np.array(tgt_aligned).astype(bool)].tolist()
        start_idx_aligned = input_data['start'][0].cpu().numpy()[np.array(tgt_aligned).astype(bool)]
        end_idx_aligned = input_data['end'][0].cpu().numpy()[np.array(tgt_aligned).astype(bool)]

        vlen = video.size(1)
        
        # method1: overlapped moving window along the time axis, then stitch
        if method == 'overlap-seq':
            eps = torch.tensor(1e-5, device=device)
            step = np.arange(0, vlen-seq_len//2, seq_len//4)
            
            # to avoid the leakage of the Ground-truth (annotated/shifted) timestamps,
            # we use the timestamps of non-alignable texts (which are their original ASR timestamps) 
            # to determine the temporal windows
            interpolate_text_mid_ts = (input_data['start'] + input_data['end'])[0].cpu().numpy() / 2

            logits = torch.zeros(len(text_str), vlen, device=device)
            overlap_counter = torch.zeros(len(text_str), vlen, device=device)
            logits_a_dual = torch.zeros(len(text_str), device=device)
            logits_a_joint = torch.zeros(len(text_str), device=device)
            text_overlap_counter = torch.zeros(len(text_str), device=device)

            for idx, step_ in enumerate(step):
                # the following line leaks GT timestamps (shown as reference, it's not used)
                # active_text_mask = np.logical_and(step_ - seq_len <= interpolate_text_mid_ts, 
                #                                   interpolate_text_mid_ts <= step_+ seq_len + seq_len)

                # avoid leaking GT timestamps (default)
                nonalignable_text_idx = np.arange(len(text_str))[~np.array(tgt_aligned).astype(bool)]
                nonalignable_text_mid_ts = interpolate_text_mid_ts[~np.array(tgt_aligned).astype(bool)]
                nonalignable_text_window_mask = np.logical_and(
                    step_ - seq_len <= nonalignable_text_mid_ts, 
                    nonalignable_text_mid_ts <= step_+ seq_len + seq_len)
                active_nonalignable_text_idx = nonalignable_text_idx[nonalignable_text_window_mask]
                if len(active_nonalignable_text_idx) == 0:
                    continue

                text_window_left, text_window_right = (
                    active_nonalignable_text_idx.min(), 
                    active_nonalignable_text_idx.max())
                active_text_mask = np.zeros((len(text_str))).astype(bool)
                # handle edge case, otherwise the heading and tailing alignable texts could be missed
                if idx <= 3:
                    text_window_left = 0
                elif idx >= len(step) - 4:
                    text_window_right = vlen
                active_text_mask[text_window_left: text_window_right+1] = True

                active_text_str = np.array(text_str)[active_text_mask].tolist()
                active_text_mask_tensor = torch.from_numpy(active_text_mask).to(device).bool()
                if np.sum(active_text_mask) == 0:
                    continue
                logits_ = get_text_visual_sim(video[:, step_:min(vlen, step_+seq_len)], active_text_str)

                if args.use_alignability_head:
                    logits_a_dual_ = logits_['alignability-dual']
                    logits_a_joint_ = logits_['alignability-joint']
                    logits_a_dual[active_text_mask_tensor] += logits_a_dual_[0,:,0]
                    logits_a_joint[active_text_mask_tensor] += logits_a_joint_[0,-1,:,0]
                    text_overlap_counter[active_text_mask_tensor] += 1
                    # logits_a_dual.append(logits_a_dual_)
                    # logits_a_joint.append(logits_a_joint_)
                else:  # if in init mode, the model is not designed for alignment task, but still we can use sim to measure alignability
                    logits_a_dual_ = logits_['dual-sim'][0,-1].max(-1).values
                    logits_a_joint_ = logits_['sim'][0,-1].max(-1).values
                    logits_a_dual[active_text_mask_tensor] += logits_a_dual_
                    logits_a_joint[active_text_mask_tensor] += logits_a_joint_
                    text_overlap_counter[active_text_mask_tensor] += 1

                logits[active_text_mask_tensor, step_:min(vlen, step_+seq_len)] += logits_['sim'][0,-1,:]
                overlap_counter[active_text_mask_tensor, step_:min(vlen, step_+seq_len)] += 1
            logits = logits.div(torch.maximum(overlap_counter, eps))
            logits_a_dual = logits_a_dual.div(torch.maximum(text_overlap_counter, eps))
            logits_a_joint = logits_a_joint.div(torch.maximum(text_overlap_counter, eps))
            sim = logits

        # method2: one pass, by interpolating the positional embedding
        elif method == 'interpolate':
            logits_ = get_text_visual_sim(video, text_str, interpolate_from=seq_len)
            sim = logits_['sim'][0,-1,:]
            if args.use_alignability_head:
                logits_a_dual = logits_['alignability-dual'][0,:,0]
                logits_a_joint = logits_['alignability-joint'][0,-1,:,0]
            else:
                logits_a_dual = logits_['dual-sim'][0,-1].max(-1).values
                logits_a_joint = logits_['sim'][0,-1].max(-1).values

        if args.use_alignability_head:
            align_score = logits_a_joint

        sim.masked_fill_(sim==0, -6e4)
        prob = sim.softmax(-1)
        vlen = sim.size(-1)

        total_align_tgt.append(np.array(tgt_aligned))
        if args.use_alignability_head:
            total_align_sim.append(align_score.cpu().numpy())
        else:
            total_align_sim.append(sim.max(-1)[0].cpu().numpy())

        sim = sim[torch.as_tensor(tgt_aligned).bool(), :]
        prob = prob[torch.as_tensor(tgt_aligned).bool(), :]

        for text_idx in range(sim.size(0)):
            s = math.floor(start_idx_aligned[text_idx])
            e = math.ceil(end_idx_aligned[text_idx])
            recall.append(s <= prob[text_idx].argmax(-1).item() <= e)

        total_vlen.append(vlen)
        total_text_count.append(len(text_str))
        total_aligned_count.append(len(text_str_aligned))

    total_align_sim = np.concatenate(total_align_sim, 0)
    total_align_tgt = np.concatenate(total_align_tgt, 0)
    assert total_align_tgt.shape == total_align_sim.shape 
    auc = metrics.roc_auc_score(total_align_tgt, total_align_sim)

    metric = {'Recall': np.mean(recall), 'AUC': auc}
    print(metric)
    return metric


def get_iou(pred, gt):
    assert pred.shape == gt.shape
    num_action = np.max(gt).astype(int)
    iou_list = []
    iod_list = []
    for act_id in range(num_action+1):
        intersect = (np.logical_and((pred==act_id), (gt==act_id))).sum()
        union = (np.logical_or((pred==act_id), (gt==act_id))).sum()
        detect = (gt==act_id).sum()
        iou = 0 if union == 0 else intersect / union
        iod = 0 if detect == 0 else intersect / detect

        iou_list.append(iou)
        iod_list.append(iod)

    if len(iou_list) == 0 or len(iod_list) == 0:
        import ipdb; ipdb.set_trace()
    return np.mean(iou_list), np.mean(iod_list)


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    sys.path.append('/work/htd/Desktop_tmp/VideoMetricLearning/process_data/feature_milnce/')
    import s3dg as milnce
    def get_word2vec_pre_projection():
        model = milnce.S3D('/work/htd/Desktop_tmp/VideoMetricLearning/process_data/feature_milnce/s3d_dict.npy', 512)
        model.load_state_dict(torch.load('/work/htd/Desktop_tmp/VideoMetricLearning/process_data/feature_milnce/s3d_howto100m.pth'))
        return model.fc

    sys.path.append('../model/')
    from word2vec_model import Word2VecTokenizer, Word2VecModel

    class DummyArgs():
        def __init__(self):
            self.tokenizer = Word2VecTokenizer()
            self.num_workers = 4
            self.model = 'align'
            self.sim = 'dot'
            self.sentence_mode = 'cls'
            self.num_encoder_layers = 0
            self.seq_len = 64
            self.use_alignability_head = False
    
    # test MILNCE raw features, with dot product
    args = DummyArgs()
    device = torch.device('cuda')
    lang_model = Word2VecModel()
    lang_model.to(device)
    milnce_fc = get_word2vec_pre_projection()
    milnce_fc.to(device)

    def get_text_visual_sim(video_embed, text_str):
        """get text-visual similarity matrix designed for S3D-word2vec model."""
        text_token = args.tokenizer(text_str, padding=True, return_tensors='pt')
        text_token = {k:v.to(device) for k,v in text_token.items()}
        text_embed = lang_model(**text_token)
        text_embed = text_embed['pooler_output']
        # mimic numpy half
        video_embed_np = video_embed.cpu().numpy().astype(np.float16)
        video_embed = torch.from_numpy(video_embed_np).float().to(device)
        v = milnce_fc(video_embed)
        return {'sim': torch.matmul(v, text_embed.transpose(0,1)).transpose(-1,-2).unsqueeze(0),
                'dual-sim': torch.matmul(v, text_embed.transpose(0,1)).transpose(-1,-2).unsqueeze(0),}

    test_alignment_htm(get_text_visual_sim, device, args)
    sys.exit(0)