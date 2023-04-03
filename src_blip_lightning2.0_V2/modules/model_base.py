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

class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

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


    def on_fit_start(self) -> None:
        print('============================ FIT LOOP ===================================')

    def on_validation_start(self) -> None:
        print('============================ VALIDATION LOOP ===================================')

    def on_test_start(self) -> None:
        print('============================ TEST LOOP ===================================')

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



