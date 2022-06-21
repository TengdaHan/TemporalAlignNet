import torch
import torch.nn as nn
from freeze_bn import freeze_bn
import sys
sys.path.append('../model')
from word2vec_model import load_milnce


class MyS3D(nn.Module):
    """A wrapper of S3D-word2vec model from https://github.com/antoine77340/S3D_HowTo100M.
    freezeBN is essential when finetuning this model."""    
    def __init__(self, language_model='word2vec', freezeBN=False, pretrained_s3d=True):
        super().__init__()
        self.freezeBN = freezeBN
        self.s3d = load_milnce(pretrained=pretrained_s3d)
        self.language_model = language_model
        if self.freezeBN:
            freeze_bn(self.s3d, 'model')

    def forward(self, inputs):
        v_feature = self.s3d(inputs)
        v_logits = self.s3d.fc(v_feature)
        return v_logits
