import numpy as np
import torch
import math
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Any, Optional, List, Dict
import gc

# from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertEmbeddings, BertModel, BertLayer
from transformers import RobertaConfig, RobertaModel

from .bert_model import BertCrossLayer, BertAttention
from .clip_model import build_model, adapt_position_encoding
from .dist_utils import concat_all_gather, all_gather_with_grad
from .blip import create_vit, init_tokenizer, load_checkpoint
from . import heads, objectives, model_utils, objective_irtr
from .med import BertConfig, BertModel
from .model_base import BaseModule

class RetrievalModule(BaseModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        vit = config['vit']
        image_size = config['image_size']
        patch_size = config['patch_size']
        vit_grad_ckpt = config['vit_grad_ckpt']
        vit_ckpt_layer = config['vit_ckpt_layer']
        med_config = '/home/tzj/codes/my_clip/src_blip_lightning/configs/med_config.json'
        self.input_text_embed_size = config['input_text_embed_size']
        self.input_image_embed_size = config['input_image_embed_size']

        self.queue_size = config['queue_size']
        self.momentum = config['momentum']
        self.distill = True

        self.visual_encoder, self.vision_width = create_vit(vit, image_size, patch_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = self.vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)
        self.text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(self.vision_width, self.input_image_embed_size)
        self.text_proj = nn.Linear(self.text_width, self.input_text_embed_size)

        self.itm_head = nn.Linear(self.text_width, 2)

        # create momentum encoders
        self.visual_encoder_m, self.vision_width = create_vit(vit, image_size)
        self.vision_proj_m = nn.Linear(self.vision_width, self.input_image_embed_size)
        self.text_encoder_m = BertModel(config=med_config, add_pooling_layer=False)
        self.text_proj_m = nn.Linear(self.text_width, self.input_text_embed_size)

        self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                            [self.vision_proj, self.vision_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            ]

        self.copy_params()

        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.negative_all_rank = config['negative_all_rank']
        # 配置评估指标
        self.set_metrics()

    # raise NotImplementedError("return tuple of train dataset class")
    def forward(self, batch, phase):
        return objective_irtr.train_irtr(self, batch, phase)

    def training_step(self, batch, batch_idx):
        # self.set_tasks()
        if self.trainer.current_epoch > 0:
            alpha = self.hparams.config['alpha']
        else:
            alpha = self.hparams.config['alpha'] * min(1, batch_idx / len(self.trainer.datamodule.train_dataloader()))
        self.hparams.config['cur_alpha'] = alpha

        irtr_loss = self(batch, phase='train')

        lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0 \
                and batch_idx % self.trainer.accumulate_grad_batches == 0:
            self.print('Global step:{global_step}.'
                       'Train Loss: {loss:.4f} '
                       'LR: {lr:.3E}'
                       .format(global_step=self.trainer.global_step,
                               loss=irtr_loss,
                               lr=lr))
        return irtr_loss

    def on_train_epoch_end(self) -> None:
        self.epoch_wrapup(phase='train')
        self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self) -> None:
        # 不传入out了，直接从self.validation_step_outputs获取每个val step的返回
        # all_preds = torch.stack(self.validation_step_outputs)
        self.epoch_wrapup(phase='val')
        self.validation_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        pass

    def on_test_epoch_end(self) -> None:
        self.epoch_wrapup(phase='test')

    # def configure_optimizers(self):
    #     opt_config = self.hparams.config['optimizer']
    #     optimizer = torch.optim.AdamW(params=self.parameters(), lr=opt_config['init_lr'],
    #                                   weight_decay=opt_config['weight_decay'])
    #
    #     # cosine_lr_schedule(optimizer, self.current_epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
    #     return {
    #         'optimizer': optimizer
    #     }

    def configure_optimizers(self):
        opt_config = self.hparams.config['optimizer']
        max_steps, warmup_steps = self.cal_steps()
        optimizer = torch.optim.AdamW(params=self.parameters(),
                                      lr=opt_config['init_lr'],
                                      weight_decay=opt_config['weight_decay'],
                                      eps=opt_config['eps'],
                                      betas=opt_config['betas'])
        sched = self.get_scheduler(optimizer, warmup_steps, max_steps)
        return {
            'optimizer': optimizer,
            'lr_scheduler': sched,
        }
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": ReduceLROnPlateau(optimizer, ...),
        #         "monitor": "metric_to_track",
        #         "frequency": "indicates how often the metric is updated"
        #         # If "monitor" references validation metrics, then "frequency" should be set to a
        #         # multiple of "trainer.check_val_every_n_epoch".
        #     },
        # }

    # patch pooling of image patches to reduce computation and enlarge receptive field
    def patch_pooling(self, x, pooled_patch_length=16):
        batch_size, seq_length, dim = x.size()
        b1 = int(np.sqrt(seq_length))
        x = x.reshape(batch_size, b1, b1, dim)
        x = x.permute(0,3,1,2)
        c1 = b1 // int(np.sqrt(pooled_patch_length))
        x = F.avg_pool2d(x, c1, stride=c1)
        x = x.permute(0,2,3,1).reshape(batch_size, pooled_patch_length, dim)
        return x

    def set_queue(self):
        # create the queue
        config = self.hparams.config
        text_len = config['max_text_len']
        image_len = (config['image_size'] // config['patch_size']) ** 2 + 1
        self.register_buffer("image_queue", torch.randn(self.input_image_embed_size, self.queue_size))
        self.register_buffer("text_queue", torch.randn(self.input_text_embed_size, self.queue_size))
        # self.register_buffer("image_embed_queue", torch.randn(self.queue_size, image_len, self.vision_width))
        self.register_buffer("text_input_ids_queue", torch.full((self.queue_size, text_len), -1))
        self.register_buffer("text_attention_mask_queue", torch.full((self.queue_size, text_len, self.text_width), -1))
        self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100))
        self.register_buffer("ptr_queue", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        self.image_embed_queue = nn.functional.normalize(self.image_embed_queue, dim=-1)

        # 太大，需要放到CPU，不用 register_buffer
        self.image_embed_queue = torch.randn(self.queue_size, image_len, self.vision_width)
    def reset_queue(self):
        self.image_queue = torch.randn_like(self.image_queue)
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.image_embed_queue = torch.randn_like(self.image_embed_queue)
        self.image_embed_queue = nn.functional.normalize(self.image_embed_queue, dim=-1)

        self.text_queue = torch.randn_like(self.text_queue)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        self.text_input_ids_queque = torch.full_like(self.text_input_ids_queque, -1)
        self.text_attention_mask_queue = torch.full_like(self.text_attention_mask_queue, -1)

        self.idx_queue = torch.full_like(self.idx_queue, -100)
        self.ptr_queue = torch.zeros_like(self.ptr_queue, dtype=torch.long)

    @classmethod
    def from_pretrained(cls, config):
        model = cls(config)
        if config['pretrained']:
            state_dict = load_checkpoint(model, config['pretrained'])
            msg = model.load_state_dict(state_dict, strict=False)
            print("missing keys:")
            print(msg.missing_keys)
        model.copy_params()
        model.set_queue()   # 清空队列，因为新增了两个队列
        return model

def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


