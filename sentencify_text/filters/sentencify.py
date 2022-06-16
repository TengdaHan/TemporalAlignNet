"""merge linebreaks, add punctuation"""

import os
from tqdm import tqdm
from glob import glob
import pandas as pd
import json
import time
import random
import numpy as np
import torch
from joblib import delayed, Parallel, parallel_backend
from transformers import AutoConfig, BertTokenizer, BertForTokenClassification
from transformers import AutoTokenizer, AutoModelForTokenClassification
# from torch.utils.data.dataloader import default_collate
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')


class Sentencify():
    def __init__(self):
        print('load tfm model')
        # self.tokenizer = AutoTokenizer.from_pretrained("felflare/bert-restore-punctuation")
        # self.model = AutoModelForTokenClassification.from_pretrained("felflare/bert-restore-punctuation")
        tokenizer_config = AutoConfig.from_pretrained(f'{os.path.dirname(__file__)}/../bert-restore-punctuation/tokenizer_config.json')
        self.tokenizer = BertTokenizer.from_pretrained(f'{os.path.dirname(__file__)}/../bert-restore-punctuation', config=tokenizer_config)
        model_config = AutoConfig.from_pretrained(f'{os.path.dirname(__file__)}/../bert-restore-punctuation/config.json')
        self.model = BertForTokenClassification.from_pretrained(f'{os.path.dirname(__file__)}/../bert-restore-punctuation/', config=model_config)
        self.label_list = ["OU", "OO", ".O", "!O", ",O", ".U", "!U", ",U", ":O", ";O", ":U", "'O", "-O", "?O", "?U"]
        self.full_stop_list = [2,3,5,6,13,14]
        self.partial_stop_list = [2,3,4,5,6,7,8,9,10,13,14]
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.device = device
        self.model.to(device)

    @torch.no_grad()
    def punctuate_and_cut(self, cap_list, start_list=None, end_list=None):
        if start_list is not None: 
            assert len(cap_list) == len(start_list) == len(end_list)
        else:
            start_list = np.zeros(len(cap_list))
            end_list = np.zeros(len(cap_list))
        # check if already punctuated
        punctuated = [(',' in i) or ('.' in i) for i in cap_list]
        punctuated_ratio = np.mean(punctuated)

        if punctuated_ratio < 0.5:
            # need punctuate
            # get timestamp per token
            token_timestamps = []
            for cap, start, end in zip(cap_list, start_list, end_list):
                cap = cap.replace(',', ' ').replace('.', ' ').replace('!', ' ').replace('?', ' ')
                cap = cap.lower()
                tokens = self.tokenizer.tokenize(cap)
                num_tokens = len(tokens)
                token_stamp = np.linspace(start, end, num_tokens+1)
                token_s = token_stamp[:-1]
                token_e = token_stamp[1::]
                token_timestamps.extend([(w,s,e) for w,s,e in 
                    zip(tokens, token_s.tolist(), token_e.tolist())])
            all_tokens = [i[0] for i in token_timestamps]
            num_tokens = len(all_tokens)
            all_token_ids = self.tokenizer.convert_tokens_to_ids(all_tokens)
            all_token_ids_batch = np.array_split(all_token_ids, num_tokens//256+1)
            all_token_ids_batch = [[101]+i.tolist()+[102] for i in all_token_ids_batch]
            max_len = max([len(i) for i in all_token_ids_batch])
            batch_size = len(all_token_ids_batch)
            # padding and convert to tensor
            input_ids = np.zeros((batch_size, max_len))
            for i in range(batch_size):
                tok = all_token_ids_batch[i]
                input_ids[i, 0:len(tok)] = tok
            input_ids = torch.from_numpy(input_ids).long().to(self.device)
            attention_mask = (input_ids != 0).long()
            punct_out = self.model(input_ids=input_ids, 
                            attention_mask=attention_mask)
            punct_prob = punct_out['logits'].softmax(-1)
            # optional: adjust the probablity of adding punctuation
            punct_prob[:,:,0:2] = punct_prob[:,:,0:2] - 0.4
            punct_pred = punct_prob.argmax(-1)
            punct_pred_list = []
            for i in range(batch_size):
                n_tok = attention_mask[i].sum()
                punct_pred_list.append(punct_pred[i, 0:n_tok][1:-1])
            punct_pred_list = torch.cat(punct_pred_list, 0)
            assert punct_pred_list.shape[0] == num_tokens

            # group to full sentence
            sentence_timestamps = []
            buffer_count = 0
            str_buffer = ''
            start_buffer = token_timestamps[0][1]
            end_buffer = token_timestamps[0][2]
            for idx, (item, pred) in enumerate(zip(token_timestamps, punct_pred_list.tolist())):
                tok = item[0]
                if tok.startswith('##'):
                    str_buffer += tok[2::]
                elif tok == "'" or str_buffer.endswith("'"):
                    str_buffer += f'{tok}'
                else:
                    str_buffer += f' {tok}'
                end_buffer = item[2]
                buffer_count += 1
                
                if idx + 1 < num_tokens and token_timestamps[idx+1][0].startswith('##'):
                    pass  # do not cut sentence on continuous token
                elif tok == "'":
                    pass  # do not cut sentence on ' symbol
                elif (buffer_count < 20 and pred in self.full_stop_list) \
                    or ((buffer_count >= 20 and pred in self.partial_stop_list)) \
                    or (idx + 1 < num_tokens and token_timestamps[idx+1][1] - item[2] > 1.0):
                    sentence_timestamps.append((str_buffer.strip(), start_buffer, end_buffer))
                    str_buffer = ''
                    buffer_count = 0
                    if idx + 1 < len(token_timestamps):
                        start_buffer = token_timestamps[idx+1][1]
                        end_buffer = token_timestamps[idx+1][2]
            if str_buffer != '':
                sentence_timestamps.append((str_buffer.strip(), start_buffer, end_buffer))

        else:
            # already punctuated
            # get timestamp per word
            word_timestamps = []
            for cap, start, end in zip(cap_list, start_list, end_list):
                words = cap.split()
                num_words = len(words)
                word_stamp = np.linspace(start, end, num_words+1)
                word_s = word_stamp[:-1]
                word_e = word_stamp[1::]
                word_timestamps.extend([(w,s,e) for w,s,e in 
                    zip(words, word_s.tolist(), word_e.tolist())])
            # group to full sentence
            sentence_timestamps = []
            str_buffer = ''
            start_buffer = word_timestamps[0][1]
            end_buffer = word_timestamps[0][2]
            for idx, item in enumerate(word_timestamps):
                str_buffer += f' {item[0]}'
                end_buffer = item[2]
                if any([i in item[0] for i in ['.', '!', '?']]):
                    sentence_timestamps.append((str_buffer.strip(), start_buffer, end_buffer))
                    str_buffer = ''
                    if idx + 1 < len(word_timestamps):
                        start_buffer = word_timestamps[idx+1][1]
                        end_buffer = word_timestamps[idx+1][2]
            if str_buffer != '':
                sentence_timestamps.append((str_buffer.strip(), start_buffer, end_buffer))
            
        # get output format 
        caps_out = [i[0] for i in sentence_timestamps]
        starts_out = [i[1] for i in sentence_timestamps]
        ends_out = [i[2] for i in sentence_timestamps]
        return caps_out, starts_out, ends_out

