import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import json
import torch
from tqdm import tqdm
from glob import glob
import ffmpeg 
import random
import math
from torch.utils.data.dataloader import default_collate
from joblib import delayed, Parallel

import sys
sys.path.append('../')
from data.loader_htm import get_htm_vlen_df


def get_vid_to_path_dict():
    """output: dict: vid --> video_path"""
    with open('vid_to_path.json') as fobj:
        content = json.load(fobj)
    return content


def check_existence(video_list, video_root, vid_to_path_dict):
    check_fn = lambda x: os.path.exists(os.path.join(video_root, vid_to_path_dict[x]))
    result = Parallel(n_jobs=8, prefer='threads')(delayed(check_fn)(i) for i in tqdm(
        video_list, total=len(video_list), desc="Check Existence", leave=False,
        disable=('/srun' in os.environ['_'])))
    existed_video = []
    for res, vid in zip(result, video_list):
        if res:
            existed_video.append(vid)
    return existed_video


class HTM_ClipLoader(Dataset):
    def __init__(self,
                 auto_align_tag='htm_aa_v1',
                 tokenizer=None,
                 mode='train',
                 num_frames=16,
                 fps=5,
                 num_sample_per_video=2,
                 center_crop_only=False):
        self.center_crop_only = center_crop_only
        self.video_root = '/scratch/shared/beegfs/shared-datasets/HowTo100M/'
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = lambda x, *args, **kwargs: {'input_ids': [0]}
        self.auto_align_tag = auto_align_tag
        self.num_frames = num_frames
        self.fps = fps
        self.num_sample_per_video = num_sample_per_video

        # loading some helper csv/json
        self.htm_vlen_df = get_htm_vlen_df()
        self.vid_to_path_dict = get_vid_to_path_dict()

        # loading HTM-AA annotations
        auto_align_anno = pd.read_csv(f'{auto_align_tag}.csv')

        # check video existence
        vid_list = list(self.vid_to_path_dict.keys())
        vid_list = check_existence(vid_list, self.video_root, self.vid_to_path_dict)

        # filter video vlen (same as MIL-NCE paper)
        proper_vlen_vids = set(self.htm_vlen_df['vid'][(self.htm_vlen_df['vlen'] < 2000) \
            & (self.htm_vlen_df['vlen'] > 64)].tolist())
        vid_list = [i for i in vid_list if i in proper_vlen_vids]

        # optionally drop problematic videos (put in list)
        drop_vid = []
        if len(drop_vid) > 0:
            vid_list = [i for i in vid_list if i not in drop_vid]

        # intersect vid_list with annotation files
        vid_list = sorted(list(set(vid_list).intersection(set(auto_align_anno['vid'].unique()))))

        # filter annotation file
        vid_set = set(vid_list)
        auto_align_anno = auto_align_anno[auto_align_anno['vid'].isin(vid_set)]

        self.vid_list = vid_list
        self.auto_align_anno = auto_align_anno

    def __len__(self):
        return len(self.vid_list)

    @staticmethod
    def collate_fn(batch):
        out_batch = {}
        key_list = list(batch[0].keys())
        for key in key_list:
            if key in ['text', 'text_idx']:
                out_batch[key] = [sample[key] for sample in batch]
            else:
                out_batch[key] = default_collate([sample[key] for sample in batch])
        return out_batch 


    def __getitem__(self, index):
        vid = self.vid_list[index]
        video_path = os.path.join(self.video_root, self.vid_to_path_dict[vid])
        vlen = self.htm_vlen_df[self.htm_vlen_df['vid'] == vid]['vlen'].values[0]
        auto_align_anno = self.auto_align_anno[self.auto_align_anno['vid'] == vid]

        # random choose text-video pair from a long video
        sample_with_replace = len(auto_align_anno) < self.num_sample_per_video
        auto_align_sample = auto_align_anno.sample(n=self.num_sample_per_video, replace=sample_with_replace).copy()

        # take text, take video
        raw_text = auto_align_sample['text'].values.tolist()
        try:
            tokens = self.tokenizer(raw_text, return_tensors='pt', padding=True)['input_ids']
        except:
            print(f'Tokenizer fails: {raw_text}, replace with [PAD]')
            tokens = torch.ones(len(raw_text), 32, dtype=torch.long) * self.tokenizer.pad_token_id

        center_timestamp = auto_align_sample['timestamp'].values.tolist()
        all_video_clips = []
        start = []
        end = []
        for ts in center_timestamp:
            v_, s_, e_ = self._get_video_frame(video_path, vlen, ts)
            all_video_clips.append(v_)
            start.append(s_)
            end.append(e_)
        all_video_clips = torch.stack(all_video_clips, dim=0)  # num,T,C,H,W

        return {'video': all_video_clips, 
                'text': raw_text, 'token': tokens, 
                'start': torch.tensor(start), 'end': torch.tensor(end), 
                'vid': vid}

    def _get_video_frame(self, video_path, vlen, timestamp):
        # modified from https://github.com/antoine77340/MIL-NCE_HowTo100M/blob/master/video_loader.py
        assert os.path.exists(video_path)
        duration = self.num_frames / self.fps  # e.g. 16/5 = 3.2s
        start = random.randint(max(0, math.floor(timestamp - duration)), 
            min(math.ceil(timestamp), vlen))
        end = start + duration

        cmd = (ffmpeg.input(video_path, ss=start, t=end-start)
                     .filter('fps', fps=self.fps)
                     .filter('pp'))
        if self.center_crop_only:
            aw, ah = 0.5, 0.5
        else:
            aw, ah = random.uniform(0,1), random.uniform(0,1)
        cmd = (cmd.crop('(iw - min(iw,ih))*{}'.format(aw),
                        '(ih - min(iw,ih))*{}'.format(ah),
                        'min(iw,ih)',
                        'min(iw,ih)').filter('scale', 224, 224))
        try:
            out, _ = (cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
                        .run(capture_stdout=True, quiet=True))
            frames = np.frombuffer(out, np.uint8).reshape([-1, 224, 224, 3]) / 255.
            frames = torch.from_numpy(frames).float()
            del out 
        except:
            print(f'failed to load video: {video_path}, replace with grey frames')
            frames = torch.ones(self.num_frames, 224, 224, 3, dtype=torch.float) * 0.5

        num_frames = frames.size(0)
        if num_frames < self.num_frames:
            zeros = torch.zeros((self.num_frames - num_frames, 224, 224, 3), dtype=torch.float)
            frames = torch.cat((frames, zeros), axis=0)
        frames = frames[0:int(self.num_frames), :]  # T,H,W,C
        frames = frames.permute(0,3,1,2)  # T,C,H,W
        return frames, start, end

