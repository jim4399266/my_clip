'''
BaseModel: 提供模型需要的一些基础方法
'''
import torch
import torch.distributed as dist
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Any, Optional, List, Dict
from transformers import RobertaConfig, RobertaModel
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

import sys
sys.path.append('..')
from gadgets.my_metrics import Accuracy, Scalar
from .dist_utils import concat_all_gather, all_gather_with_grad
from .blip import create_vit, init_tokenizer, load_checkpoint
from .objective_irtr import val_irtr, recall_eval

class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def on_fit_start(self) -> None:
        print('============================ FIT LOOP ===================================')
        self.training_step_outputs = []


    def on_validation_start(self) -> None:
        print('============================ VALIDATION LOOP ===================================')
        self.validation_step_outputs = []

    def on_test_start(self) -> None:
        print('============================ TEST LOOP ===================================')
        self.test_step_outputs = []

    @classmethod
    def from_pretrained(cls, config):
        model = cls(config)
        if config['pretrained']:
            state_dict = load_checkpoint(model, config['pretrained'])
            msg = model.load_state_dict(state_dict, strict=False)
            print("missing keys:")
            print(msg.missing_keys)
        model.copy_params()
        return model

    def freeze_module(self, module):
        """
        Freezes module's parameters.
        """
        for parameter in module.parameters():
            parameter.requires_grad = False
    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    def cal_steps(self):
        # 计算 max_steps 和 warmup_steps
        if self.trainer.max_steps == None or self.trainer.max_epochs != None:
            max_steps = (len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs
                         // self.hparams.config['gradient_accumulation_steps'])
        else:
            max_steps = self.trainer.max_steps
        # 当 warmup_steps=-1 时不启用warm up
        warmup_steps = max(0, self.hparams.config['warmup_steps'])
        if isinstance(warmup_steps, float):
            warmup_steps = int(warmup_steps * max_steps)
        print(f'====== Max steps: {max_steps},\t Warm up steps: {warmup_steps} =========')
        return max_steps, warmup_steps

    def get_scheduler(self, optimizer, warmup_steps, max_steps):
        # 设置scheduler
        if self.hparams.config['scheduler'] == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps,
            )
        elif self.hparams.config['scheduler'] == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps,
                num_cycles=self.hparams.config['num_cycles']
            )
        else:
            scheduler = None
        sched = {
            'scheduler': scheduler, 'interval': 'step'
        }
        return sched
    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idxs):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat, self.trainer.world_size)
        text_feats = concat_all_gather(text_feat, self.trainer.world_size)

        batch_size = image_feats.shape[0]

        ptr = int(self.ptr_queue)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.ptr_queue[0] = ptr

    def set_metrics(self):
        # 区分训练集和验证集的指标
        for split in ['train', 'val', 'test']:
            for k, v in self.hparams.config['task_name'].items():
                if v < 1:
                    continue
                if k == 'irtr':
                    setattr(self, f"{split}_{k}_loss", Scalar())
                elif k == 'itm':
                    setattr(self, f"{split}_{k}_accuracy", Accuracy())
                    setattr(self, f"{split}_{k}_loss", Scalar())

    def set_tasks(self):
        self.current_tasks = [k for k, v in self.hparams.config['task_name'].items() if v >= 1]

    def epoch_wrapup(self, phase):
        the_metric = 0
        total_loss = 0
        if 'irtr' in self.hparams.config['task_name'] and not self.training:
            if phase == 'val':
                data_loader = self.trainer.datamodule.val_dataloader()
            else:
                data_loader = self.trainer.datamodule.test_dataloader()
            score_val_i2t, score_val_t2i = val_irtr(self, data_loader)
            val_result = recall_eval(score_val_i2t, score_val_t2i, data_loader.dataset.index_mapper)
            print(f'global_step: {self.global_step}')
            print(val_result)
            for item in ['txt_r1', 'txt_r5', 'txt_r10', 'txt_r_mean', 'img_r1', 'img_r5', 'img_r10', 'img_r_mean',
                         'r_mean']:
                self.logger.experiment.add_scalar(
                    f"{phase}/irtr/{item}", val_result[item], self.global_step
                )

            the_metric += (val_result['txt_r1'] + val_result['img_r1']) * 10 \
                          + (val_result['txt_r5'] + val_result['img_r5']) * 5 \
                          + val_result['txt_r10'] + val_result['img_r10']
            self.log(f'{phase}/irtr/the_metric', the_metric)
            self.log(f'{phase}/irtr/r_mean', val_result['r_mean'])





