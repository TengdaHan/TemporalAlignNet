import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import simplejson as json 
sys.path.append('../model')
from word2vec_model import Word2VecTokenizer


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


def pad_sequence_to_size(sequences, size, batch_first=True, padding_value=0):
    shape = list(sequences[0].shape)
    shape[0] = size
    dummy = torch.zeros(shape, device=sequences[0].device)
    pad_dummy = pad_sequence([dummy]+sequences, 
                             batch_first=batch_first, 
                             padding_value=padding_value)
    if batch_first:
        pad_out = pad_dummy[1::]
    else:
        pad_out = pad_dummy[:,1::]
    return pad_out


def get_holdout_set():
    with open(os.path.join(os.path.dirname(__file__), 'htm_holdout_vid.txt')) as f:
        holdout_vids = f.readlines()
        holdout_vids_set = set([i.strip() for i in holdout_vids])
    return holdout_vids_set


def get_htm_vlen_df():
    htm_vlen_df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), 'htm_vlen.csv'),
        names=['vid','vlen']
    )
    return htm_vlen_df


def get_vid_to_asr_dict(json_path):
    """output: dict: vid --> csv path for ASR sentences"""
    with open(json_path) as fobj:
        content = json.load(fobj)
    return content


class HTM_FeatureLoader():
    def __init__(self,
                 text_tag='htm-fe',
                 tokenizer=None,
                 mode='train',
                 duration=64,
                 trim_ratio=0.1,
                 ):
        self.video_feature_path = '/scratch/shared/beegfs/shared-datasets/HowTo100M/howto100m_s3d_features'
        self.text_tag = text_tag
        self.mode = mode
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = lambda x: {'input_ids': [0]}
        self.duration = duration
        self.trim_ratio = trim_ratio  # not used for now

        # loading some helper csv/json
        tag_to_asr = {
            'htm-fe': 'vid_to_asr_fe.json',
            'htm-370k': 'sentencified_htm_370k.json',
            'htm-1200k': 'sentencified_htm_1200k.json'}
        holdout_vids_set = get_holdout_set()
        self.htm_vlen_df = get_htm_vlen_df()
        self.vid_to_asr_dict = get_vid_to_asr_dict(
            os.path.join(os.path.dirname(__file__), tag_to_asr[text_tag]))
        
        all_vids = list(self.vid_to_asr_dict.keys())

        # remove vids from heldout set
        all_vids = [i for i in all_vids if i not in holdout_vids_set]

        # filter video vlen (same as MIL-NCE paper)
        proper_vlen_vids = set(self.htm_vlen_df['vid'][(self.htm_vlen_df['vlen'] < 1000) \
            & (self.htm_vlen_df['vlen'] > 64)].tolist())
        all_vids = [i for i in all_vids if i in proper_vlen_vids]
        all_vids = sorted(all_vids)

        # because training data is enough, we use first 5% (cap at 1000 samples) as val set
        num_val = min(int(len(all_vids) * 0.05), 1000) 
        if mode == 'train':
            self.video_info = all_vids[num_val::]
        elif mode in ['val', 'test']:
            self.video_info = all_vids[0:num_val]

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
        if 'cut_start' in batch[0]:
            out_batch['cut_start'] = [sample['cut_start'] for sample in batch]
        if 'cut_end' in batch[0]:
            out_batch['cut_end'] = [sample['cut_end'] for sample in batch]
        if 'abs_text_start' in batch[0]:
            out_batch['abs_text_start'] = [sample['abs_text_start'] for sample in batch]
        if 'abs_text_end' in batch[0]:
            out_batch['abs_text_end'] = [sample['abs_text_end'] for sample in batch]
        return out_batch 

    def __getitem__(self, index):
        # we first sample some text, get the start/end timestamps,
        # then cut the video clip based on the start/end timestamps
        vid = self.video_info[index]

        # load feature first, check vlen
        try:
            path = os.path.join(self.video_feature_path, "{}.mp4.npy".format(vid))
            array = np.load(path)
            feature = torch.from_numpy(array)
        except:
            path = os.path.join(self.video_feature_path, "{}.webm.npy".format(vid))
            array = np.load(path)
            feature = torch.from_numpy(array)
        del array
        vlen = feature.size(0)

        # sample text, then cut the video
        caps, (start_timestamp, end_timestamp) = self._get_text(vid, vlen)
        video_feature = self._get_video_feature(feature, vid, start_timestamp, end_timestamp)
        video_padding_mask = torch.zeros(video_feature.size(0)).long()

        # check paddings for batch operation
        if isinstance(self.tokenizer, Word2VecTokenizer):
            caps['token'] = torch.stack(caps['token'], 0)
        else:
            caps['token'] = pad_sequence_to_size(caps['token'], size=32, batch_first=True, padding_value=0)

        output_dict = {'video': video_feature,
                'padding_mask': video_padding_mask,
                'vid': vid,
                'text': caps['text'],
                'start': caps['start'],
                'end': caps['end'],
                'token': caps['token'],
                'abs_text_start': (np.array(caps['start']).astype(np.float32) + start_timestamp) / vlen,
                'abs_text_end': (np.array(caps['end']).astype(np.float32) + start_timestamp) / vlen,
                }
        # also give video start/end for visualization
        if self.mode in ['val', 'test']:
            output_dict = {**output_dict, 'cut_start': start_timestamp, 'cut_end': end_timestamp}
        return output_dict


    def _get_text(self, vid, vlen):
        if self.text_tag in ['htm-370k', 'htm-1200k']:
            cap_df = pd.DataFrame.from_dict(self.vid_to_asr_dict[vid])
        else:
            cap_df = pd.read_csv(self.vid_to_asr_dict[vid])
        cap_df = cap_df[cap_df['end'] < vlen]

        no_caption_flag = False

        if len(cap_df['end'].tolist()):
            last_timestamp = cap_df['end'].tolist()[-1]
        else:
            no_caption_flag = True
        
        if not no_caption_flag:
            if (cap_df['start'] < last_timestamp - self.duration - 1).sum() == 0:
                no_caption_flag = True
            else:
                start_idx = np.random.choice(
                    cap_df.index[cap_df['start'] < last_timestamp - self.duration])
                start_timestamp = int(round(cap_df.iloc[start_idx]['start']))
                end_timestamp = start_timestamp + self.duration

        sentences = []
        tokens = []
        starts = []
        ends = []

        if not no_caption_flag:
            for idx in range(start_idx, len(cap_df)):
                cap_entry = cap_df.iloc[idx]
                text, s, e = cap_entry['text'], cap_entry['start'], cap_entry['end']
                s, e = round(s), round(e)
                text = str(text).replace('\n',' ').strip()
                if len(text.split()) > 256:
                    text = ' '.join(text.split()[0:256])
                if s > end_timestamp or e-s < 1:
                    break
                elif e > end_timestamp:
                    e = end_timestamp

                token = self.tokenizer(text, max_length=32, truncation=True)['input_ids']
                trim_start = max(s - start_timestamp, 0)
                trim_end = min(e - start_timestamp, self.duration)
                if trim_end == trim_start:
                    break

                if isinstance(self.tokenizer, Word2VecTokenizer) and (sum(token) == 0):  # all words are stop words
                    break

                sentences.append(text)
                tokens.append(torch.tensor(token))
                starts.append(trim_start)
                ends.append(trim_end)

        if len(sentences) == 0 or no_caption_flag:  # handle unlucky sampling
            text = '[UNK]'
            token_ = torch.tensor(self.tokenizer(text)['input_ids'])
            tokens.append(token_)
            sentences.append(text)
            starts.append(0)
            ends.append(self.duration)
            if no_caption_flag:
                start_timestamp = 0
                end_timestamp = self.duration

        return {'text': sentences, 'start': starts, 'end': ends, 'token': tokens,}, \
                (start_timestamp, end_timestamp)


    def _get_video_feature(self, feature, vid, start, end):
        try:
            feature_cut = feature[start:end, :]
        except:
            feature_cut = feature[start::, :]
            tmp = feature_cut[-1].unsqueeze(0).repeat(self.duration, 1)
            tmp[0:feature_cut.shape[0], :] = feature_cut
            feature_cut = tmp

        # for debugging
        if feature_cut.size(0) == 0: 
            print(f'Error log: {vid} with shape {feature.shape} {start}-{end}, is size 0')

        return feature_cut.float()



if __name__ == '__main__':
    tokenizer = Word2VecTokenizer()
    D = HTM_FeatureLoader(tokenizer=tokenizer)
    loader = torch.utils.data.DataLoader(D, batch_size=4, num_workers=0,
        collate_fn=D.collate_fn)
    
    for output in tqdm(loader, total=len(loader)):
        print(output.keys())
        break
