import copy
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

class BLIPModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        vit = config['vit']
        image_size = config['image_size']
        vit_grad_ckpt = config['vit_grad_ckpt']
        vit_ckpt_layer = config['vit_ckpt_layer']
        med_config = '/home/tzj/codes/my_clip/src_blip_lightning/configs/med_config.json'
        embed_dim = 256

        self.queue_size = config['queue_size']
        self.momentum = config['momentum']

        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)

        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2)

        # create momentum encoders
        self.visual_encoder_m, vision_width = create_vit(vit, image_size)
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel(config=med_config, add_pooling_layer=False)
        self.text_proj_m = nn.Linear(text_width, embed_dim)

        self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                            [self.vision_proj, self.vision_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            ]
        # TODO 是否需要换位置
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100))
        self.register_buffer("ptr_queue", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)


        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.negative_all_rank = config['negative_all_rank']
        # 配置评估指标
        model_utils.set_metrics(self)

        if config['pretrained']:
            state_dict = load_checkpoint(self, config['pretrained'])
            msg = self.load_state_dict(state_dict, strict=False)
            print("missing keys:")
            print(msg.missing_keys)


    def forward(self, batch, phase):
        return objective_irtr.train_irtr(self, batch, phase)

    def on_fit_start(self) -> None:
        print('============================ FIT LOOP ===================================')

    def on_train_epoch_start(self) -> None:
        config = self.hparams.config
        model_utils.cosine_lr_schedule(self.trainer.optimizers[0], self.current_epoch, config['max_epoch'], config['init_lr'], config['min_lr'])

    def training_step(self, batch, batch_idx):
        model_utils.set_tasks(self)
        if self.trainer.current_epoch > 0:
            alpha = self.hparams.config['alpha']
        else:
            alpha = self.hparams.config['alpha'] * min(1, batch_idx / len(self.trainer.datamodule.train_dataloader()))
        self.hparams.config['cur_alpha'] = alpha

        irtr_loss = self(batch, phase='train')
        return irtr_loss

    def on_train_epoch_end(self) -> None:
        model_utils.epoch_wrapup(self, phase='train')

    def on_validation_start(self) -> None:
        print('============================ VALIDATION LOOP ===================================')

    def validation_step(self, batch, batch_idx):
        pass
        # model_utils.set_tasks(self)
        # # 如果只有检索任务，则跳过
        # if len(self.current_tasks) == 1 and self.hparams.config['loss_name'].get('irtr', 0) > 0:
        #     return None
        # output = self(batch, phase='val')
        # return output

    def on_validation_epoch_end(self) -> None:
        # 不传入out了，直接从self.validation_step_outputs获取每个val step的返回
        # all_preds = torch.stack(self.validation_step_outputs)
        model_utils.epoch_wrapup(self, phase='val')

    def on_test_start(self) -> None:
        print('============================ TEST LOOP ===================================')

    def test_step(self, batch, batch_idx):
        pass
        # model_utils.set_tasks(self)
        # # 如果只有检索任务，则跳过
        # if len(self.current_tasks) == 1 and self.hparams.config['loss_name'].get('irtr', 0) > 0:
        #     return None
        # output = self(batch, phase='test')
        # return output

    def on_test_epoch_end(self) -> None:
        model_utils.epoch_wrapup(self, phase='test')

    def configure_optimizers(self):
        config = self.hparams.config
        optimizer = torch.optim.AdamW(params=self.parameters(), lr=config['init_lr'],
                                      weight_decay=config['weight_decay'])

        # cosine_lr_schedule(optimizer, self.current_epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
        return {
            'optimizer': optimizer
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

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

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




