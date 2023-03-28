'''
检索任务
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import glob
import json
from tqdm import tqdm
import functools

from torch.utils.data.distributed import DistributedSampler

from .dist_utils import all_gather
# 将缓存的张量转为Dataset

def irtr_encoder(pl_module, dataloader):
    # ======================================== 预先对数据进行编码 ========================================
    device = pl_module.device
    ret = dict()
    for batch in tqdm(dataloader, ncols=80, desc='Getting embedding...'):
        batch['image'] = batch['image'].to(device)
        batch['text_encodings'] = {k: v.to(device) for k, v in batch['text_encodings'].items()}
        batch['image_index'] = batch['image_index'].view(-1)
        batch['text_list_index'] = batch['text_list_index'].view(-1)
        text_ret = pl_module.text_encoder(batch)
        image_ret = pl_module.image_encoder(batch)

        for k, v in text_ret.items():
            if v != None:
                ret[k] = torch.cat([ret.get(k, torch.tensor([], device='cpu')), v.cpu()], dim=0)
        for k, v in image_ret.items():
            if v != None:
                ret[k] = torch.cat([ret.get(k, torch.tensor([], device='cpu')), v.cpu()], dim=0)
    # ======================================== 所有向量处理完毕 ========================================
    return ret

def trans_dot_recall(pl_module, encoder_ret):
    # 点积，用于对比学习
    text_cls, image_cls, tiids, iids = (
        encoder_ret['text_cls'].cpu(), encoder_ret['image_cls'].cpu(),
        encoder_ret['tiids'].cpu(), encoder_ret['iids'].cpu()
    )
    # 计算所有图片和所有文本的相似度
    image_logits = pl_module.itc_logits(image_cls, text_cls, scaling=False)
    text_logits = pl_module.itc_logits(text_cls, image_cls, scaling=False)
    return image_logits.tolist(), text_logits.tolist(), iids.tolist(), tiids.tolist()

def compute_irtr_recall_(pl_module, dataloader):
    encoder_ret = irtr_encoder(pl_module, dataloader)
    image_logits, text_logits, rank_iids, tiids = trans_dot_recall(pl_module, encoder_ret)
    # # cat output
    # if 'cat' in pl_module.hparams.config['representation_fn']:
    #     image_logits, text_logits, rank_iids, tiids = trans_cat_recall(pl_module, encoder_ret)
    # # dot output
    # else:
    #     image_logits, text_logits, rank_iids, tiids = trans_dot_recall(pl_module, encoder_ret)

    return image_logits, text_logits, rank_iids, tiids
