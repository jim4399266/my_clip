import copy
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Any, Optional, List, Dict
import gc

from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertEmbeddings, BertModel, BertLayer
from transformers import RobertaConfig, RobertaModel

from .bert_model import BertCrossLayer, BertAttention
from .clip_model import build_model, adapt_position_encoding
from .dist_utils import concat_all_gather

from . import heads, objectives, model_utils

class ModelModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.text_backbone, self.text_backbone_config = model_utils.get_text_backbone(config)
        self.image_backbone = model_utils.get_image_backbone(config)

        # self.token_type_embeddings = nn.Embedding(2, config['hidden_size'])
        # self.token_type_embeddings.apply(objectives.init_weights)

        self.cross_modal_text_transform = nn.Linear(config['input_text_embed_size'], config['hidden_size'])
        self.cross_modal_text_transform.apply(objectives.init_weights)
        self.cross_modal_image_transform = nn.Linear(config['input_image_embed_size'], config['hidden_size'])
        self.cross_modal_image_transform.apply(objectives.init_weights)

        # 两个模态的交互层，每个模态各有 num_top_layer 层
        # self.cross_modal_image_layers = nn.ModuleList(
        #     [BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])]
        # )
        # self.cross_modal_image_layers.apply(objectives.init_weights)
        # self.cross_modal_text_layers = nn.ModuleList(
        #     [BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])]
        # )
        # self.cross_modal_text_layers.apply(objectives.init_weights)

        # Pooler池化层
        # self.cross_modal_image_pooler = heads.Pooler(config['hidden_size'])
        # self.cross_modal_image_pooler.apply(objectives.init_weights)
        # self.cross_modal_text_pooler = heads.Pooler(config['hidden_size'])
        # self.cross_modal_text_pooler.apply(objectives.init_weights)

        # create the queue
        self.register_buffer("image_queue", torch.randn(config['hidden_size'], config['queue_size']))
        self.register_buffer("text_queue", torch.randn(config['hidden_size'], config['queue_size']))
        self.register_buffer("idx_queue", torch.full((1, config['queue_size']), -100))
        self.register_buffer("ptr_queue", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.queue_size = config['queue_size']

        # 根据预训练目标任务选择Head
        # if config['loss_name']['itm'] > 0:
        #     self.itm_logits = heads.ITMHead(config['hidden_size'] * 2) # 因为需要输入两个模态
        #     self.itm_logits.apply(objectives.init_weights)
        # 根据下游目标任务选择Head
        if config['loss_name']['irtr'] > 0:
            # 对两个模态的cls进行点积计算相似度
            self.itc_logits = heads.ITCHead()
            self.itc_logits.apply(objectives.init_weights)
        # 配置评估指标
        model_utils.set_metrics(self)
        self.current_tasks = list()
        # ===================== load downstream ======================
        # 也可以在trainer中添加ckpt： trainer.test(model, datamodule=dm, ckpt_path=str(ckpt))
        if config['load_path'] != "":
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if not config['test_only']:
                state_dict = adapt_position_encoding(state_dict, after=self.hparams.config['image_size'],
                                                 patch_size=self.hparams.config['patch_size'])
            self.load_state_dict(state_dict, strict=False)
            print(f'Load model weights from:{config["load_path"]}')
            del state_dict

        # self.freeze()

    def freeze(self):
        model_utils.freeze_module(self.text_backbone.embeddings)
        freeze_layers = len(self.text_backbone.encoder.layer) // 2
        model_utils.freeze_module(self.text_backbone.encoder.layer[:freeze_layers])

        # model_utils.freeze_module(self.image_backbone.token_embedding)
        # freeze_layers = len(self.image_backbone.encoder.layer) // 2
        # model_utils.freeze_module(self.image_backbone.encoder.layer[:freeze_layers])

    def text_encoder(self, batch, mask_text=False):
        do_mlm = "_mlm" if mask_text else ""
        text_labels = batch.get(f"text_labels{do_mlm}", None)
        tiids = batch.get('text_list_index', None)

        text_encodings = batch.get(f"text_encodings{do_mlm}", None)
        input_ids = text_encodings.get("input_ids", None)
        text_masks = text_encodings.get("attention_mask", None)
        # 文本编码
        text_embeds = self.text_backbone(input_ids=input_ids, attention_mask=text_masks)[0]
        text_cls = self.cross_modal_text_transform(text_embeds[:, 0, :])
        text_extend_masks = self.text_backbone.get_extended_attention_mask(
            text_masks, text_masks.size(), device=self.device)
        return {
            'text_embeds': text_embeds,
            'text_labels': text_labels,
            'text_masks': text_masks,
            'text_extend_masks': text_extend_masks,
            'text_cls': text_cls,
            'tiids': tiids.long()
        }

    def image_encoder(self, batch, mask_image=False, image_token_type_idx=1, img=None, iids=None):
        img = batch.get('image', None)
        iids = batch.get('image_index', None)
        # 图像编码
        # vit 将图片分成patch，比如图片大小是224，patch大小 32
        # 则一共分成 (224 / 32)^2 + 1 个patch，多出来的1是cls标志
        image_embeds = self.image_backbone(img)
        image_cls = self.cross_modal_image_transform(image_embeds[:, 0, :])
        image_masks = torch.ones((image_embeds.size(0), image_embeds.size(1)), dtype=torch.long, device=self.device)
        image_extend_masks = self.text_backbone.get_extended_attention_mask(image_masks, image_masks.size(), device=self.device)
        # TODO 作者根据ViLT的方法，加上 token_type_embedding
        # image_embeds = image_embeds + self.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx))
        # image_cls = self.cross_modal_image_pooler(image_embeds)
        return {
            'image_embeds': image_embeds,
            'image_masks': image_masks,
            'image_extend_masks': image_extend_masks,
            'image_cls': image_cls,
            'iids': iids.long(),
        }

    def forward(self, batch, phase):
        self.itc_logits.clamp_scaler()
        ret, encoder_ret = dict(), dict()
        encoder_ret.update(self.text_encoder(batch))
        encoder_ret.update(self.image_encoder(batch))

        # # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            # ret.update(objectives.compute_irtr(self, encoder_ret, phase))
            ret.update(objectives.compute_irtr_q(self, encoder_ret, phase))

        # # Image Text Matching
        # if "itm" in self.current_tasks:
        #     ret.update(objectives.compute_itm(self, encoder_ret, phase))
        return ret

    def training_step(self, batch, batch_idx):
        model_utils.set_tasks(self)
        output = self(batch, phase='train')
        total_loss = sum([v for k, v in output.items() if 'loss' in k])
        return total_loss

    def training_epoch_end(self, outs):
        model_utils.epoch_wrapup(self, phase='train')

    def validation_step(self, batch, batch_idx):
        model_utils.set_tasks(self)
        # 如果只有检索任务，则跳过
        if len(self.current_tasks) == 1 and self.hparams.config['loss_name'].get('irtr', 0) > 0:
            return None
        output = self(batch, phase='val')
        return output

    def validation_epoch_end(self, outs):
        model_utils.epoch_wrapup(self, phase='val')

    def test_step(self, batch, batch_idx):
        model_utils.set_tasks(self)
        # 如果只有检索任务，则跳过
        if len(self.current_tasks) == 1 and self.hparams.config['loss_name'].get('irtr', 0) > 0:
            return None
        output = self(batch, phase='test')
        return output

    def test_epoch_end(self, outs):
        model_utils.epoch_wrapup(self, phase='test')

    def configure_optimizers(self):
        return model_utils.set_schedule(self)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idx, world_size=1):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat, world_size=world_size)
        text_feats = concat_all_gather(text_feat, world_size=world_size)
        idxs = concat_all_gather(idx, world_size=world_size)

        assert image_feats.shape[0] == text_feats.shape[0]
        assert image_feats.shape[0] == idxs.shape[0]

        batch_size = image_feats.shape[0]
        ptr = int(self.ptr_queue)
        # assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        # self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        # self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        # self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        # ptr = (ptr + batch_size) % self.queue_size  # move pointer
        # 数据集最后一个batch大小可能不是正好等于batch_size
        if ptr + batch_size <= self.queue_size:
            self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
            self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
            self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
            ptr = (ptr + batch_size) % self.queue_size  # move pointer
        else:
            t = (ptr + batch_size) - self.queue_size  # 超出了的部分舍弃
            self.image_queue[:, ptr:ptr + batch_size] = image_feats[:-t].T
            self.text_queue[:, ptr:ptr + batch_size] = text_feats[:-t].T
            self.idx_queue[:, ptr:ptr + batch_size] = idxs[:-t].T
            ptr = 0  # reset pointer
        self.ptr_queue[0] = ptr