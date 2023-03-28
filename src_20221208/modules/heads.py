import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union

from transformers.models.bert.modeling_bert import BertPredictionHeadTransform

class Pooler(nn.Module):
    '''
    对cls token进行映射
    '''
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        cls_token_tensor = hidden_states[:, 0]
        pooled_output = self.activation(self.dense(cls_token_tensor))
        return pooled_output

class FeedForward(nn.Module):
    '''
    text transformer 和 vision transformer 都用的LayerNorm
    '''
    def __init__(self, input_size, output_size):
        super(FeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size)
        self.dense = nn.Linear(input_size, output_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        output = self.activation(hidden_states)
        return output

class ImageFeedForward(nn.Module):
    def __init__(self, num_features, input_size, output_size):
        super(ImageFeedForward, self).__init__()
        self.batch_norm = nn.BatchNorm2d(num_features)
        self.dense = nn.Linear(input_size, output_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        output = self.activation(hidden_states)
        return output

class ITCHead(nn.Module):
    '''
    用于irtr对比学习任务 (Image-Text Contrastive) ，对不同模态的向量直接进行点积计算相似度
    '''
    def __init__(self, logit_scale:Union[int, float, nn.Parameter]=None):
        super(ITCHead, self).__init__()
        # self.logit_scale = logit_scale if logit_scale != None else nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale = 100

    def forward(self, features_A, features_B, normalized=False):
        if isinstance(self.logit_scale, nn.Parameter):
            logit_scale = self.logit_scale.to(features_A.device)
        else:
            logit_scale = self.logit_scale
        if not normalized:
            # 保证后两个维度一致，因为归一化需要对应相同纬度
            assert features_A.shape[-1] == features_B.shape[-1]
            # features_A = features_A / features_A.norm(dim=-1, keepdim=True)
            # features_B = features_B / features_B.norm(dim=-1, keepdim=True)
            features_A = F.normalize(features_A, dim=-1)
            features_B = F.normalize(features_B, dim=-1)
        logits_A = logit_scale * features_A @ features_B.T
        return logits_A