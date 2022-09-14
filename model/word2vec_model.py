import os
import torch
from torch import nn
import torch.nn.functional as F
import re
import numpy as np
import sys
from s3d_milnce.s3dg import S3D

def load_milnce(pretrained=True):
    model = S3D(os.path.join(os.path.dirname(__file__), 's3d_milnce/s3d_dict.npy'), 512)
    if pretrained:
        model.load_state_dict(torch.load(
            os.path.join(os.path.dirname(__file__), 's3d_milnce/s3d_howto100m.pth')))
    return model

def get_word2vec_text_module():
    model = load_milnce()
    return model.text_module

def get_word2vec_pre_projection():
    model = load_milnce()
    return model.fc, model.text_module


class Word2VecTokenizer():
    def __init__(self, max_words=32):
        text_module = get_word2vec_text_module()
        self.word_to_token = text_module.word_to_token
        self.token_to_word = {v:k for k,v in self.word_to_token.items()}
        self.max_words = max_words

    def _split_sentence(self, sentence):
        w = re.findall(r"[\w']+", str(sentence).lower())
        return w

    def _words_to_token(self, words):
        """word(str) to token(int), pad or cut to max_words"""
        ids = []
        for idx, w in enumerate(words):
            if idx >= self.max_words:
                break
            try:
                ids.append(self.word_to_token[w])
            except:
                ids.append(0)
        if len(ids) < self.max_words:
            ids = ids + [0] * (self.max_words - len(ids))
        return ids[0:self.max_words]

    def tokenize(self, inputs):
        """sentence to [str, ],
        or [sentence, ] to [[str, ], [str, ]]"""
        if isinstance(inputs, str):
            return self._split_sentence(inputs)
        elif isinstance(inputs, list):
            return [self._split_sentence(i) for i in inputs]

    def __call__(self, inputs, padding=True, return_tensors=None, **kwargs):
        assert padding, f"padding = {padding} is not supported" 
        if isinstance(inputs, list):
            inputs_token = [self._words_to_token(self._split_sentence(sent.lower())) for sent in inputs]
        elif isinstance(inputs, str):
            inputs_token = self._words_to_token(self._split_sentence(inputs.lower()))
        
        if return_tensors == 'pt':
            attention_mask = (np.array(inputs_token) != 0).astype(np.uint8)
            return {'input_ids': torch.from_numpy(np.array(inputs_token)),
                    'attention_mask': torch.from_numpy(attention_mask)}
        elif return_tensors is None:
            attention_mask = (np.array(inputs_token) != 0).astype(np.uint8).tolist()
            return {'input_ids': inputs_token,
                    'attention_mask': attention_mask}


class Word2VecModel(nn.Module):
    def __init__(self):
        super().__init__()
        text_module = get_word2vec_text_module()
        self.word_embd = text_module.word_embd
        self.fc1 = text_module.fc1
        self.fc2 = text_module.fc2
        
    def forward(self, input_ids, attention_mask=None, *args, **kwargs):
        with torch.no_grad():
            x = self.word_embd(input_ids)
        x = F.relu(self.fc1(x), inplace=True)

        # original implementation from Miech et al.:
        # x = F.relu(self.fc1(x), inplace=True)
        # x = torch.max(x, dim=1)[0]
        # x = self.fc2(x)
        # we modified so it supports ignoring certain positions i.e. padding tokens
        if attention_mask is not None:  # 1 means keep, 0 means ignore
            attention_mask[(attention_mask.sum(-1) == 0),:] = True  # in case the whole sentence is all stop words
            x_ = x.masked_fill(~attention_mask[:,:,None].bool(), -6e4)
            pooled_output = torch.max(x_, dim=1).values
        else:
            pooled_output = torch.max(x, dim=1).values

        return {'last_hidden_state': self.fc2(x), 
                'pooler_output': self.fc2(pooled_output)}


if __name__ == '__main__':
    model = Word2VecModel()
    tokenizer = Word2VecTokenizer()

    token = tokenizer(["hello world"], return_tensors='pt')
    out = model(**token)

    import ipdb; ipdb.set_trace()
