'''
对应不同任务的处理方式
'''

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
from einops import rearrange

from .dist_utils import all_gather
from .objectives_irtr import compute_irtr_recall_

def softmax(logits, t=0.01):
    return nn.functional.softmax(logits / t)

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias != None:
            module.bias.data.zero_()
    elif isinstance(module, (nn.LayerNorm)):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

def compute_irtr(pl_module, encoder_ret, phase):
    # batch 内的对比学习，对应的图文为正样本，其余为负样本
    labels = torch.arange(len(encoder_ret['iids']), dtype=torch.long, device=pl_module.device)
    # infer 输出经过text_encoder 和 image_encoder 得到的向量
    text_cls, image_cls = encoder_ret['text_cls'], encoder_ret['image_cls']
    logits_per_text = pl_module.itc_logits(text_cls, image_cls)
    logits_per_image = pl_module.itc_logits(image_cls, text_cls)

    image_loss = F.cross_entropy(logits_per_image, labels)
    text_loss = F.cross_entropy(logits_per_text, labels)
    irtr_loss = (image_loss + text_loss) / 2
    ret = {'irtr_loss': irtr_loss}
    irtr_loss = getattr(pl_module, f"{phase}_irtr_loss")(ret["irtr_loss"])
    pl_module.log(f"irtr/{phase}/irtr_loss", irtr_loss)
    return ret

# def compute_irtr(pl_module, encoder_ret, phase):
#     # batch 内的对比学习，对应的图文为正样本，其余为负样本
#     labels = torch.arange(len(encoder_ret['iids']), dtype=torch.long, device=pl_module.device)
#     labels1 = torch.cat([torch.eye(len(labels)), torch.zeros(len(labels), 200)], dim=-1).to(pl_module.device)
#     # infer 输出经过text_encoder 和 image_encoder 得到的向量
#     text_cls, image_cls = encoder_ret['text_cls'], encoder_ret['image_cls']
#     logits_per_text = pl_module.itc_logits(text_cls, image_cls)
#     logits_per_image = pl_module.itc_logits(image_cls, text_cls)
#
#     logits_per_text1, logits_per_image1 = torch.cat([logits_per_text, torch.ones(len(labels), 200).to(pl_module.device)], dim=-1),\
#         torch.cat([logits_per_image, torch.ones(len(labels), 200).to(pl_module.device)], dim=-1)
#
#     image_loss = F.cross_entropy(logits_per_image, labels)
#     text_loss = F.cross_entropy(logits_per_text, labels)
#     # image_loss1 = F.cross_entropy(logits_per_image1, labels1)
#     # text_loss1 = F.cross_entropy(logits_per_text1, labels1)
#     irtr_loss = (image_loss + text_loss) / 2
#     # ret = {'irtr_loss': irtr_loss}
#     # irtr_loss = getattr(pl_module, f"{phase}_irtr_loss")(ret["irtr_loss"])
#     # pl_module.log(f"irtr/{phase}/irtr_loss", irtr_loss)
#     # return ret
#     return {}

def compute_irtr_q(pl_module, encoder_ret, phase):
    # 添加队列扩充对比样本量，对应的图文为正样本，其余为负样本
    world_size = pl_module.hparams.config['num_gpus']
    # 将当前batch的label与队列拼接
    labels = encoder_ret['iids'].view(-1, 1)
    labels_all = torch.cat([labels.t(), pl_module.idx_queue.clone().detach()], dim=1)
    pos_idx = torch.eq(labels, labels_all).float()
    sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

    # infer 输出经过text_encoder 和 image_encoder 得到的向量
    text_cls, image_cls = encoder_ret['text_cls'], encoder_ret['image_cls']
    text_feat_all = torch.cat([text_cls.T, pl_module.text_queue.clone().detach()], dim=1)
    image_feat_all = torch.cat([image_cls.T, pl_module.image_queue.clone().detach()], dim=1)

    logits_per_text = pl_module.itc_logits(text_cls, image_feat_all.T)
    logits_per_image = pl_module.itc_logits(image_cls, text_feat_all.T)

    image_loss = F.cross_entropy(logits_per_image, sim_targets)
    text_loss = F.cross_entropy(logits_per_text, sim_targets)
    # loss_i2t = -torch.sum(F.log_softmax(logits_per_image, dim=1) * sim_targets, dim=1).mean()
    # loss_t2i = -torch.sum(F.log_softmax(logits_per_text, dim=1) * sim_targets, dim=1).mean()

    irtr_loss = (image_loss + text_loss) / 2
    ret = {'irtr_loss': irtr_loss}

    pl_module._dequeue_and_enqueue(image_cls, text_cls, labels, world_size)

    irtr_loss = getattr(pl_module, f"{phase}_irtr_loss")(ret["irtr_loss"])
    pl_module.log(f"irtr/{phase}/irtr_loss", irtr_loss)
    return ret

@torch.no_grad()
def compute_irtr_recall(pl_module, phase):
    # 加载数据集
    if phase == 'val':
        dataloader = pl_module.trainer.datamodule.val_dataloader()
    else:
        dataloader = pl_module.trainer.datamodule.test_dataloader()

    # 获得图文相似度排序以及对应的id
    image_logits, text_logits, rank_iids, rank_tiids = compute_irtr_recall_(pl_module, dataloader)
    del dataloader

    iids = torch.tensor(rank_iids).view(-1)     # image_logits 图片的id: [ size_i ]
    tiids = torch.tensor(rank_tiids).view(-1)   # text_logits 文本的id（文本对应图片的id）: [ size_t ]
    image_logits = torch.tensor(image_logits).view(len(iids), -1) # [ size_i, size_t ]
    text_logits = torch.tensor(text_logits).view(len(tiids), -1)  # [ size_t, size_i ]
    print(f'------- {phase} data size: image {len(iids)}, text {len(tiids)}')

    # tr: image2text
    # topk: 每张图片取最相关的 k 个文本
    # scores: [ image_size, text_size ] --(dim=1)-->  topk: [ image_size, k ]
    tr_top10 = image_logits.topk(10, dim=1, largest=True, sorted=True)
    tr_top5 = image_logits.topk(5, dim=1, largest=True, sorted=True)
    tr_top1 = image_logits.topk(1, dim=1, largest=True, sorted=True)
    # 此时 tr_topk 带有indices 和 values 两个属性，需要找出对应文本的id
    tr_top10_ids = tiids[tr_top10.indices]
    tr_top5_ids = tiids[tr_top5.indices]
    tr_top1_ids = tiids[tr_top1.indices]
    # 如果前k个中有匹配的，则得1分（max=1），计算所有样本的平均得分
    tr_r10 = (iids.unsqueeze(1) == tr_top10_ids).float().max(dim=1).values.mean()
    tr_r5 = (iids.unsqueeze(1) == tr_top5_ids).float().max(dim=1).values.mean()
    tr_r1 = (iids.unsqueeze(1) == tr_top1_ids).float().max(dim=1).values.mean()

    # ir:  text2image
    # topk: 每个文本取最相关的topk张图片
    # scores:[text_size, image_size] --(dim=1)-->    topk: [text_size, k]
    ir_top10 = text_logits.topk(10, dim=1, largest=True, sorted=True)
    ir_top5 = text_logits.topk(5, dim=1, largest=True, sorted=True)
    ir_top1 = text_logits.topk(1, dim=1, largest=True, sorted=True)
    ir_top10_ids = iids[ir_top10.indices]
    ir_top5_ids = iids[ir_top5.indices]
    ir_top1_ids = iids[ir_top1.indices]
    ir_r10 = (tiids.unsqueeze(1) == ir_top10_ids).float().max(dim=1).values.mean()
    ir_r5 = (tiids.unsqueeze(1) == ir_top5_ids).float().max(dim=1).values.mean()
    ir_r1 = (tiids.unsqueeze(1) == ir_top1_ids).float().max(dim=1).values.mean()
    torch.cuda.empty_cache()
    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)

