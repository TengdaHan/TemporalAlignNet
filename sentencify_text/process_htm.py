import os
from tqdm import tqdm
from glob import glob
import pandas as pd
import json
import time
import random
import numpy as np
from joblib import delayed, Parallel
from torch.utils.data.dataloader import default_collate
from filters.utils import filter_length, filter_language, merge_linebreaks
from filters.sentencify import Sentencify
import torch


def master_filter(cap_list):
    """if pass all filtering conditions, return True"""
    return int(filter_length(cap_list) and filter_language(cap_list))


@torch.no_grad()
def master_process(cap_dict):
    """remove duplicates, then cut into sentences"""
    cap_list, start_list, end_list = cap_dict['text'], cap_dict['start'], cap_dict['end'],
    caps, starts, ends = merge_linebreaks(cap_list, start_list, end_list)
    if len(caps) <= 5:
        return None
    caps, starts, ends = processor.punctuate_and_cut(caps, starts, ends)
    return {'text': caps, 'start': starts, 'end': ends}


# for pre-fetching
class CapLoader():
    def __init__(self, raw_caption_json):
        tic = time.time()
        print(f'loading raw caption from json: {raw_caption_json}')
        with open(raw_caption_json) as f:
            raw_caption = json.load(f)
        print(f'loading completed, takes {time.time()-tic} seconds')
        self.raw_caption = raw_caption

        valid_vids_txt = f'tmp/passed_vids_{raw_caption_json.split("_")[-1][0:-5]}.txt'
        with open(valid_vids_txt, 'r') as fr:
            valid_vids = fr.readlines()
            valid_vids = [i.strip() for i in valid_vids]
        self.valid_vids = valid_vids

    @staticmethod
    def collate_fn(batch):
        out = []
        out.append(default_collate([sample[0] for sample in batch]))
        out.append([sample[1] for sample in batch])
        return out

    def __getitem__(self, index):
        vid = self.valid_vids[index]
        cap_dict = self.raw_caption[vid]
        cap_list, start_list, end_list = cap_dict['text'], cap_dict['start'], cap_dict['end']
        caps, starts, ends = merge_linebreaks(cap_list, start_list, end_list)

        if len(caps) <= 5:
            return vid, None
        else:
            return vid, (caps, starts, ends)

    def __len__(self):
        return len(self.valid_vids)

NUM_CHUNKS = 8

if __name__ == '__main__':
    htm_root = 'your_path/HowTo100M'

    ### Step 1 ### 
    ### split huge HowTo100M json file into chunks for easy processing
    raw_caption_json = f'{htm_root}/raw_caption.json'
    tic = time.time()
    print('loading raw caption from json')
    with open(raw_caption_json) as f:
        raw_caption = json.load(f)
    print(f'loading completed, takes {time.time()-tic} seconds')

    os.makedirs('chunks', exist_ok=True)
    vid_keys = sorted(list(raw_caption.keys()))
    vid_keys_chunks = np.array_split(vid_keys, NUM_CHUNKS)
    for idx, chunk in enumerate(vid_keys_chunks):
        tmp_json = {k:raw_caption[k] for k in chunk}
        out_path = f'chunks/raw_caption_chunk{idx+1:02d}.json'
        if not os.path.exists(out_path):
            with open(out_path, 'w') as fw:
                json.dump(tmp_json, fw, indent=2)
        else:
            print(f'chunk {idx+1}/{NUM_CHUNKS} file existed')
        print(f'chunk {idx+1}/{NUM_CHUNKS} completed')

    ### Step 2 ### 
    ### apply language filter and length filter, save output to tmp/
    os.makedirs('tmp', exist_ok=True)
    all_chunks = [
        'chunks/raw_caption_chunk01.json',
        'chunks/raw_caption_chunk02.json',
        'chunks/raw_caption_chunk03.json',
        'chunks/raw_caption_chunk04.json',
        'chunks/raw_caption_chunk05.json',
        'chunks/raw_caption_chunk06.json',
        'chunks/raw_caption_chunk07.json',
        'chunks/raw_caption_chunk08.json',
        ]

    for raw_caption_json in all_chunks:
        tic = time.time()
        print(f'loading raw caption from json: {raw_caption_json}')
        with open(raw_caption_json) as f:
            raw_caption = json.load(f)
        print(f'loading completed, takes {time.time()-tic} seconds')

        vid_keys = list(raw_caption.keys())
        output = Parallel(n_jobs=16, prefer='processes')(delayed(master_filter)(raw_caption[i]['text']) for i in tqdm(vid_keys, total=len(vid_keys)))
        print('filter pass ratio:', np.mean(output))
        passed_vids = np.array(vid_keys)[np.array(output).astype(bool)]
        with open(f'tmp/passed_vids_{raw_caption_json.split("_")[-1][0:-5]}.txt', 'w') as fw:
            fw.write('\n'.join(passed_vids.tolist()))

    ### Step 3 ###
    ### use Sentencify module to get sentences
    processor = Sentencify()
    avg_sentence_length_all = 0
    os.makedirs('output', exist_ok=True)
    
    with torch.no_grad():
        for raw_caption_json in all_chunks:
            D = CapLoader(raw_caption_json)
            loader = torch.utils.data.DataLoader(D, batch_size=1,
                shuffle=False, num_workers=8, collate_fn=D.collate_fn)

            valid_vids = []
            output = []
            for idx, (vid, item) in tqdm(enumerate(loader), total=len(loader)):
                vid = vid[0]
                item = item[0]
                if item is None:
                    continue
                caps, starts, ends = item
                caps_, starts_, ends_ = processor.punctuate_and_cut(caps, starts, ends)
                output.append({'text': caps_, 'start': starts_, 'end': ends_})
                valid_vids.append(vid)

                avg_sentence_length = np.mean([len(i.split(' ')) for i in caps_])
                avg_sentence_length_all += avg_sentence_length
                if idx % 100 == 0:
                    print(f'avg sent length = {avg_sentence_length_all/(idx+1):.2f}')

            save_dict = {k:v for k,v in zip(valid_vids, output) if v is not None}
            with open(f'output/punct_raw_caption_{raw_caption_json.split("_")[-1][0:-5]}.json', 
                        'w') as fw:
                json.dump(save_dict, fw, indent=2)
