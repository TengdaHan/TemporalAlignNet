from pandas.core.frame import DataFrame
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import random
import time
import re
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate
import json 
import lmdb
import math
from loader_htm import pad_sequence_by_last

disk = 'beegfs'

# TODO: Not fully implemented


class YouCook2_DataLoader(Dataset):
    def __init__(self, 
                 mode='train',
                 source='feature_milnce_len16_16fps',
                 tokenizer=None,
                 ):
        mode_ = 'val' if mode == 'test' else mode
        self.mode = mode 
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = lambda x: {'input_ids': [0]}
        anno_path = '/scratch/shared/beegfs/htd/DATA/YouCook2'

        with open(os.path.join(anno_path, 
            f'feat_csv_vid2path_{mode_}.json')) as fp:
            self.vid2path = json.load(fp)

        # get split 
        self.video_feature_path = os.path.join(anno_path, f'{source}/{mode_}')

        vlen_path = os.path.join(anno_path, 
            f'splits/{mode_}_duration_totalframe.csv')
        vlen_df = pd.read_csv(vlen_path)
        vlen_dict = {vid:[duration,num_frame] \
                     for vid,duration,num_frame in zip(
                         vlen_df['vid_id'], vlen_df['duration'], vlen_df['total_frame'])}
        self.vlen_dict = vlen_dict
        video_info = sorted(list(self.vlen_dict.keys()))

        # check existence
        video_info = [i for i in video_info \
                      if os.path.exists(
                          os.path.join(anno_path, source, mode_, 
                              f"{self.vid2path[i].split('/')[-2]}_{i}.pth.tar"))]

        # drop certain features because different vlen
        drop_list = ['FtHLUsOntqI', 'HQtOXHghaL0', 'ffoRmenLSLs', 'wKHC2gbRdA0']
        self.video_info = [i for i in video_info if i not in drop_list]

        if mode == 'val':
            random.seed(0)
            self.video_info = random.sample(
                self.video_info, int(len(self.video_info)//2)) # for fast eval

        print((f'YouCook2 feature loader get {len(video_info)}'
               f'->{len(self.video_info)} videos '
               f'from {self.video_feature_path}'))

        with open(os.path.join(anno_path, 
            'youcookii_annotations_trainval.json')) as fp:
            anno = json.load(fp)
        self.anno = anno['database']

    def __len__(self):
        return len(self.video_info)

    