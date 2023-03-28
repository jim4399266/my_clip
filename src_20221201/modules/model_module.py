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

from . import heads, objectives, model_utils

class ModelModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.is_clip = (not 'swin' in config['vit'])

        if 'roberta' in config['tokenizer']:
            Config = RobertaConfig
        else:
            Config = BertConfig
        bert_config = Config(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        resolution_after = config['image_size']

        self.token_type_embeddings = nn.Embedding(2, config['hidden_size'])
        self.token_type_embeddings.apply(objectives.init_weights)

        # 两个模态的编码器，使用预训练好的模型
        if self.is_clip:
            self.vit_model = build_model(config['vit'], resolution_after=resolution_after)

        if 'roberta' in config['tokenizer']:
            self.text_transformer = RobertaModel.from_pretrained(config['tokenizer'])
        else:
            self.text_transformer = BertModel.from_pretrained(config['tokenizer'])

        self.cross_modal_text_transform = nn.Linear(config['input_text_embed_size'], config['hidden_size'])
        self.cross_modal_text_transform.apply(objectives.init_weights)
        self.cross_modal_image_transform = nn.Linear(config['input_image_embed_size'], config['hidden_size'])
        self.cross_modal_image_transform.apply(objectives.init_weights)

        # 两个模态的交互层，每个模态各有 num_top_layer 层
        self.cross_modal_image_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])]
        )
        self.cross_modal_image_layers.apply(objectives.init_weights)
        self.cross_modal_text_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])]
        )
        self.cross_modal_text_layers.apply(objectives.init_weights)

        # Pooler池化层
        self.cross_modal_image_pooler = heads.Pooler(config['hidden_size'])
        self.cross_modal_image_pooler.apply(objectives.init_weights)
        self.cross_modal_text_pooler = heads.Pooler(config['hidden_size'])
        self.cross_modal_text_pooler.apply(objectives.init_weights)

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
        if config['load_path'] != "":
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if not config['test_only']:
                state_dict = adapt_position_encoding(state_dict, after=resolution_after,
                                                 patch_size=self.hparams.config['patch_size'])
            self.load_state_dict(state_dict, strict=False)
            print(f'Load model weights from:{config["load_path"]}')
            del state_dict

    def text_encoder(self, batch, mask_text=False):
        do_mlm = "_mlm" if mask_text else ""
        text_encodings = batch.get(f"text_encodings{do_mlm}", None)
        text_labels = batch.get(f"text_labels{do_mlm}", None)
        tiids = batch.get('text_list_index', None)
        input_ids = text_encodings.get("input_ids", None)
        text_masks = text_encodings.get("attention_mask", None)
        # 文本编码
        text_embeds = self.text_transformer(input_ids=input_ids, attention_mask=text_masks)[0]
        text_embeds = self.cross_modal_text_transform(text_embeds)
        text_extend_masks = self.text_transformer.get_extended_attention_mask(
            text_masks, text_masks.size(), device=self.device)
        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks))
        text_cls = self.cross_modal_text_pooler(text_embeds)
        return {
            'text_embeds': text_embeds,
            'text_labels': text_labels,
            'text_masks': text_masks,
            'text_extend_masks': text_extend_masks,
            'text_cls': text_cls,
            'tiids': tiids.long()
        }

    def image_encoder(self, batch, mask_image=False, image_token_type_idx=1, img=None, iids=None):
        if img is None:
            if f"image_{image_token_type_idx - 1}" in batch:
                imgkey = f"image_{image_token_type_idx - 1}"
            else:
                imgkey = "image"
            img = batch[imgkey]
        iids = batch.get('image_index', None)
        # 图像编码
        # vit 将图片分成patch，比如图片大小是224，patch大小 32
        # 则一共分成 (224 / 32)^2 + 1 个patch，多出来的1是cls标志
        image_embeds = self.vit_model(img)
        image_embeds = self.cross_modal_image_transform(image_embeds)
        device = image_embeds.device
        image_masks = torch.ones((image_embeds.size(0), image_embeds.size(1)), dtype=torch.long, device=device)
        image_extend_masks = self.text_transformer.get_extended_attention_mask(image_masks, image_masks.size(), device=device)
        image_embeds = image_embeds + self.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx))
        image_cls = self.cross_modal_image_pooler(image_embeds)
        return {
            'image_embeds': image_embeds,
            'image_masks': image_masks,
            'image_extend_masks': image_extend_masks,
            'image_cls': image_cls,
            'iids': iids.long(),
        }


    def forward(self, batch, phase):
        ret, encoder_ret = dict(), dict()
        encoder_ret.update(self.text_encoder(batch))
        encoder_ret.update(self.image_encoder(batch))

        # # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, encoder_ret, phase))

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
        output = self(batch, phase='val')
        return output

    def validation_epoch_end(self, outs):
        model_utils.epoch_wrapup(self, phase='val')

    def test_step(self, batch, batch_idx):
        model_utils.set_tasks(self)
        output = self(batch, phase='test')
        return output

    def test_epoch_end(self, outs):
        model_utils.epoch_wrapup(self, phase='test')

    def configure_optimizers(self):
        return model_utils.set_schedule(self)