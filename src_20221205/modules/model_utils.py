'''
构建模型的一些工具
'''
import torch
import random
import os
from transformers.optimization import AdamW
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

import sys
sys.path.append('..')
from gadgets.my_metrics import Accuracy, Scalar
from .dist_utils import all_gather
from .objectives import compute_irtr_recall
from .clip_model import build_model

def get_text_backbone(config, backbone_config_path=None, pretrained=True):
    # 加载模型配置文件
    if backbone_config_path and os.path.isdir(backbone_config_path):
        backbone_config = AutoConfig.from_pretrained(backbone_config_path, output_hidden_states=True)
    elif backbone_config_path and os.path.isfile(backbone_config_path):
        backbone_config = torch.load(backbone_config_path)
    else:
        backbone_config = AutoConfig.from_pretrained(config['tokenizer'], output_hidden_states=True)
    # 加载预训练模型
    if pretrained:
        backbone = AutoModel.from_pretrained(config['tokenizer'], config=backbone_config)
    else:
        backbone = AutoModel.from_config(backbone_config)
    return backbone, backbone_config

def get_image_backbone(config):
    is_clip = (not 'swin' in config['vit'])
    # 两个模态的编码器，使用预训练好的模型
    if is_clip:
        image_backbone = build_model(config['vit'], resolution_after=config['image_size'])
    return image_backbone

def freeze_module(module):
    """
    Freezes module's parameters.
    """
    for parameter in module.parameters():
        parameter.requires_grad = False

def set_metrics(pl_module):
    # 区分训练集和验证集的指标
    for split in ['train', 'val', 'test']:
        for k, v in pl_module.hparams.config['loss_name'].items():
            if v < 1:
                continue
            if k == 'irtr':
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == 'itm':
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())

def set_tasks(pl_module):
    pl_module.current_tasks = [k for k, v in pl_module.hparams.config['loss_name'].items() if v >=1]

def set_schedule(pl_module):
    config = pl_module.hparams.config
    lr = config['learning_rate']
    wd = config['weight_decay']
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'norm.bias', 'norm.weight', 'norm1.bias', 'norm1.weight', 'norm2.bias', 'norm2.weight']
    head_names = ['mlm_logits', 'itm_logits', 'itc_logits']
    cross_modal_names = ['cross_modal']
    lr_mult_head = config['lr_mult_head']
    lr_mult_cross_modal = config['lr_mult_cross_modal']
    end_lr = config['end_lr']
    decay_power = config['decay_power']
    optim_type = config['optim_type'].lower()

    # any() 函数用于判断给定的可迭代参数 iterable 是否全部为 False，则返回 False，如果有一个为 True，则返回 True。
    optimizer_grouped_parameters = [
        {
            # 不属于no_decay、head_names、cross_modal_names的参数
            # weight decay和learning rate按照默认值
            'name': 'transformer',
            'names': [
                n for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)  # n里不包含no_decay的任何字段
                   and not any(bb in n for bb in head_names)  # n里不包含head_names的任何字段
                   and not any(ht in n for ht in cross_modal_names)  # n里不包含cross_modal_names的任何字段
            ],
            'params': [
                p for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)  # n里不包含no_decay的任何字段
                   and not any(bb in n for bb in head_names)  # n里不包含head_names的任何字段
                   and not any(ht in n for ht in cross_modal_names)  # n里不包含cross_modal_names的任何字段
            ],
            # 下层预训练模型用较小的lr
            'weight_decay': wd,
            'lr': lr,

        },
        {
            # 属于no_decay，不属于head_names、cross_modal_names的参数
            # weight decay为 0，learning rate按照默认值
            'name': 'transformer w/o wd',
            'names': [
                n for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                   and not any(bb in n for bb in head_names)
                   and not any(ht in n for ht in cross_modal_names)
            ],
            'params': [
                p for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                   and not any(bb in n for bb in head_names)
                   and not any(ht in n for ht in cross_modal_names)
            ],
            'weight_decay': 0.0,
            'lr': lr,
        },
        {
            # 属于head_names，不属于no_decay、cross_modal_names的参数
            # weight decay按照默认值，learning rate需要乘上系数 lr_mult_head
            'name': 'heads',
            'names': [
                n for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                   and any(bb in n for bb in head_names)
                   and not any(ht in n for ht in cross_modal_names)
            ],
            'params': [
                p for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                   and any(bb in n for bb in head_names)
                   and not any(ht in n for ht in cross_modal_names)
            ],
            # 上层 head 用较大的lr
            'weight_decay': wd,
            'lr': lr * lr_mult_head,

        },
        {
            # 属于no_decay、head_names，不属于cross_modal_names的参数
            # weight decay为 0，learning rate需要乘上系数 lr_mult_head
            'name': 'heads w/o wd',
            'names': [
                n for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                   and any(bb in n for bb in head_names)
                   and not any(ht in n for ht in cross_modal_names)
            ],
            'params': [
                p for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                   and any(bb in n for bb in head_names)
                   and not any(ht in n for ht in cross_modal_names)
            ],
            'weight_decay': 0.0,
            'lr': lr * lr_mult_head,
        },
        {
            # 属于cross_modal_names，不属于no_decay、head_names的参数
            # weight decay按照默认值，learning rate需要乘上系数 lr_mult_cross_modal
            'name': 'cross_modal layers',
            'names': [
                n for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                   and not any(bb in n for bb in head_names)
                   and any(ht in n for ht in cross_modal_names)
            ],
            'params': [
                p for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                   and not any(bb in n for bb in head_names)
                   and any(ht in n for ht in cross_modal_names)
            ],
            # 上层 cross_modal层 用较大的lr
            'weight_decay': wd,
            'lr': lr * lr_mult_cross_modal,

        },
        {
            # no_decay、属于cross_modal_names，不属于head_names的参数
            # weight decay为 0，learning rate需要乘上系数 lr_mult_cross_modal
            'name': 'cross_modal layers w/o wd',
            'names': [
                n for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                   and not any(bb in n for bb in head_names)
                   and any(ht in n for ht in cross_modal_names)
            ],
            'params': [
                p for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                   and not any(bb in n for bb in head_names)
                   and any(ht in n for ht in cross_modal_names)
            ],
            'weight_decay': 0.0,
            'lr': lr * lr_mult_cross_modal,
        }
    ]

    if optim_type == 'adamw':
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=config['eps'], betas=config['betas']
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)
    else:
        optimizer = None

    if pl_module.trainer.max_steps == None or pl_module.trainer.max_epochs != None:
        max_steps = max(1,
            len(pl_module.trainer.datamodule.train_dataloader()) * pl_module.trainer.max_epochs
            // pl_module.trainer.accumulate_grad_batches
        )
    else:
        max_steps = pl_module.trainer.max_steps

    warmup_steps = config['warmup_steps']
    if isinstance(warmup_steps, float):
        warmup_steps = int(warmup_steps * max_steps)

    # pl_module.max_steps = max_steps
    # pl_module.warmup_steps = warmup_steps
    print(f'Max steps: {max_steps}, warmup steps: {warmup_steps}')
    if decay_power == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps,lr_end=end_lr, power=decay_power
        )
    sched = {
        'scheduler': scheduler,
        'interval': 'step'
    }
    return ([optimizer], [sched])

def epoch_wrapup(pl_module, phase):
    the_metric = 0
    total_loss = 0

    if pl_module.hparams.config['get_recall_metric'] and not pl_module.training:
        # 检索需要单独计算
        ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10 = compute_irtr_recall(pl_module, phase)
        print(f'global_step: {pl_module.global_step}\n'
              f'ir_r1: {ir_r1*100:.2f} %, ir_r5: {ir_r5*100:.2f} %, ir_r10: {ir_r10*100:.2f} %, '
              f'tr_r1: {tr_r1*100:.2f} %, tr_r5: {tr_r5*100:.2f} %, tr_r10: {tr_r10*100:.2f} %.')
        for item in ['ir_r1', 'ir_r5', 'ir_r10', 'tr_r1', 'tr_r5', 'tr_r10']:
            pl_module.logger.experiment.add_scalar(
                f"{phase}/recalls/{item}", eval(item), pl_module.global_step
            )

        the_metric += (ir_r1.item() + tr_r1.item()) * 10 \
                      + (ir_r5.item() + tr_r5.item()) * 5 \
                      + ir_r10.item() + tr_r10.item()

    for loss_name, v in pl_module.hparams.config['loss_name'].items():
        if v < 1:
            continue
        value = 0
        if loss_name == 'irtr':
            loss = getattr(pl_module, f'{phase}_{loss_name}_loss').compute()
            pl_module.log(f'{loss_name}/{phase}/loss_epoch', loss)
            getattr(pl_module, f'{phase}_{loss_name}_loss').reset()
        elif loss_name == 'itm':
            value = getattr(pl_module, f'{phase}_{loss_name}_accuracy', 0).compute()
            pl_module.log(f'{loss_name}/{phase}/accuracy_epoch', value)
            getattr(pl_module, f'{phase}_{loss_name}_accuracy').reset()
            loss = getattr(pl_module, f'{phase}_{loss_name}_loss').compute()
            pl_module.log(f'{loss_name}/{phase}/loss_epoch', loss)
            getattr(pl_module, f'{phase}_{loss_name}_loss').reset()
        the_metric += value
        # total_loss += loss
    pl_module.log(f'{phase}/the_metric', the_metric)
    # pl_module.log(f'{phase}/total_loss', total_loss)
