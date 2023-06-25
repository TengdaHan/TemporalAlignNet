import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate
import json 
import lmdb
import math
import spacy

from model.word2vec_model import Word2VecTokenizer

def pad_sequence_by_last(sequences):
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    out_dims = (len(sequences), max_len) + trailing_dims
    out_tensor = sequences[0].new_full(out_dims, 0.0)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length, ...] = tensor 
        out_tensor[i, length:, ...] = tensor[-1, ...]
    return out_tensor

disk = 'beegfs'

def get_htm_vlen_df():
    htm_vlen_df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), 'htm_vlen.csv'),
        names=['vid','vlen']
    )
    return htm_vlen_df


class HTM_Align(Dataset):
    def __init__(self,
                 tokenizer=None,
                 mode='val',
                 duration=64,
                 use_spacy_pos=False):
        self.video_feature_path = '/scratch/shared/beegfs/htd/DATA/HowTo100M/howto100m_s3d_features'
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = lambda x: {'input_ids': [0]}
        self.duration = duration

        htm_vlen_df = get_htm_vlen_df()
        
        source='aligned_htm.json'
        anno_path = f'{os.path.dirname(os.path.abspath(__file__))}/../data/{source}'
        with open(anno_path) as fp: anno = json.load(fp)
        self.anno = anno
        for i in self.anno.keys():
            assert os.path.exists(os.path.join(self.video_feature_path, "{}.mp4.npy".format(i)))
        self.video_info = sorted(self.anno.keys())

    def __len__(self):
        return len(self.video_info)

    @staticmethod
    def collate_fn(batch):
        out_batch = {}
        out_batch['video'] = pad_sequence_by_last([sample['video'] for sample in batch])
        out_batch['padding_mask'] = pad_sequence([sample['padding_mask'] for sample in batch], batch_first=True, padding_value=1.0)
        out_batch['text'] = [sample['text'] for sample in batch]
        out_batch['start'] = [sample['start'] for sample in batch]
        out_batch['end'] = [sample['end'] for sample in batch]
        out_batch['vid'] = [sample['vid'] for sample in batch]
        out_batch['token'] = [sample['token'] for sample in batch]
        out_batch['pos'] = [sample['pos'] for sample in batch]
        out_batch['align'] = [sample['align'] for sample in batch]
        return out_batch 

    def __getitem__(self, idx):
        vid = self.video_info[idx]
        anno = self.anno[vid]
        caps, starts, ends, text_aligned = [],[],[],[]
        for seg in anno:
            text_aligned.append(seg[0])
            starts.append(seg[1])
            ends.append(seg[2])
            caps.append(seg[3])

        last_timestamp = ends[-1]
        cap_df = pd.DataFrame.from_dict(
            {'text': caps,
            'start': starts,
            'end': ends,
            'aligned': text_aligned}
        )

        del caps, starts, ends, text_aligned

        start_idx = np.random.choice(
            cap_df.index[cap_df['start'] < last_timestamp - self.duration])
        start_timestamp = int(math.ceil(cap_df.iloc[start_idx]['start']))
        end_timestamp = start_timestamp + self.duration
        
        sentences = []
        tokens = []
        starts = []
        ends = []
        align_flag = []
        pos_flag = []

        for idx in range(start_idx, len(cap_df)):
            cap_entry = cap_df.iloc[idx]
            text, s, e, aligned = cap_entry['text'], cap_entry['start'], \
                cap_entry['end'], cap_entry['aligned']
            s, e = round(s), round(e)
            text = text.replace('\n',' ').strip()
            if len(text.split()) > 256:
                text = ' '.join(text.split()[0:256])
            if s > end_timestamp or e-s < 1:
                break
            elif e > end_timestamp:
                e = end_timestamp

            token = self.tokenizer(text)['input_ids']
            
            if isinstance(self.tokenizer, Word2VecTokenizer):
                token_pos = np.array(token) != 0
            else:
                token_pos = [0,1,0] # placeholder, no effect

            if np.sum(token_pos) == 0:
                break

            sentences.append(text)
            tokens.append(torch.tensor(token))
            pos_flag.append(torch.tensor(token_pos, dtype=torch.long))
            starts.append(max(s - start_timestamp, 0))
            ends.append(min(e - start_timestamp, self.duration))
            align_flag.append(aligned)

        # video 
        path = os.path.join(self.video_feature_path, "{}.mp4.npy".format(vid))
        array = np.load(path)
        feature = torch.from_numpy(array)
        try:
            feature_cut = feature[start_timestamp:end_timestamp, :]
        except:
            feature_cut = feature[start_timestamp::, :]
            tmp = feature_cut[-1].unsqueeze(0).repeat(self.duration, 1)
            tmp[0:feature_cut.shape[0], :] = feature_cut
            feature_cut = tmp
        video_feature = feature_cut.float()
        video_padding_mask = torch.zeros(video_feature.size(0)).long()

        return {
            'video': video_feature,
            'padding_mask': video_padding_mask,
            'vid': vid,
            'text': sentences,
            'start': starts,
            'end': ends,
            'token': tokens,
            'pos': pos_flag,
            'align': align_flag,
        }