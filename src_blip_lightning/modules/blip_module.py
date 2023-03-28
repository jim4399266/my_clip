import copy
import torch
import math
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Any, Optional, List, Dict
import gc

# from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertEmbeddings, BertModel, BertLayer
from transformers import RobertaConfig, RobertaModel

from .bert_model import BertCrossLayer, BertAttention
from .clip_model import build_model, adapt_position_encoding
from .dist_utils import concat_all_gather
from .blip import create_vit, init_tokenizer, load_checkpoint
from . import heads, objectives, model_utils
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
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        image = batch['image']
        caption = batch['text']
        alpha = self.hparams.config['cur_alpha']
        idx = batch['image_index']

        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=35,
                              return_tensors="pt").to(image.device)

        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                        return_dict=True, mode='text')
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

        ###============== Image-text Contrastive Learning ===================###
        idx = idx.view(-1, 1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image)
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
            image_feat_m_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)

            text_output_m = self.text_encoder_m(text.input_ids, attention_mask=text.attention_mask,
                                                return_dict=True, mode='text')
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
            text_feat_m_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            sim_i2t_m = image_feat_m @ text_feat_m_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_m_all / self.temp

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = image_feat @ text_feat_m_all / self.temp
        sim_t2i = text_feat @ image_feat_m_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2

        idxs = concat_all_gather(idx, world_size=self.trainer.world_size)
        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idxs)

        ###============== Image-text Matching ===================###
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id

        # forward the positve image-text pair         # 正相关的图文对进行融合，得到output_pos，是正相关图文对融合后的向量
        bs = image.size(0)
        output_pos = self.text_encoder(encoder_input_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       )

        if self.negative_all_rank:   # 如果是分布式，从所有卡中抽取负样本
            # compute sample similarity
            with torch.no_grad():
                mask = torch.eq(idx, idxs.t())

                image_feat_world = concat_all_gather(image_feat, self.trainer.world_size)
                text_feat_world = concat_all_gather(text_feat, self.trainer.world_size)

                sim_i2t = image_feat @ text_feat_world.t() / self.temp
                sim_t2i = text_feat @ image_feat_world.t() / self.temp

                weights_i2t = F.softmax(sim_i2t, dim=1)
                weights_i2t.masked_fill_(mask, 0)

                weights_t2i = F.softmax(sim_t2i, dim=1)
                weights_t2i.masked_fill_(mask, 0)

            image_embeds_world = all_gather_with_grad(image_embeds, self.trainer.world_size)

            # select a negative image (from all ranks) for each text
            image_embeds_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds_world[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

            # select a negative text (from all ranks) for each image
            input_ids_world = concat_all_gather(encoder_input_ids, self.trainer.world_size)
            att_mask_world = concat_all_gather(text.attention_mask, self.trainer.world_size)

            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(input_ids_world[neg_idx])
                text_atts_neg.append(att_mask_world[neg_idx])

        else:  # 仅从当前卡上的批次中抽取负样本
            with torch.no_grad():
                mask = torch.eq(idx, idx.t())

                sim_i2t = image_feat @ text_feat.t() / self.temp
                sim_t2i = text_feat @ image_feat.t() / self.temp

                weights_i2t = F.softmax(sim_i2t, dim=1)
                weights_i2t.masked_fill_(mask, 0)

                weights_t2i = F.softmax(sim_t2i, dim=1)
                weights_t2i.masked_fill_(mask, 0)

                # select a negative image (from same rank) for each text
            image_embeds_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

            # select a negative text (from same rank) for each image
            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(encoder_input_ids[neg_idx])
                text_atts_neg.append(text.attention_mask[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)
        # 这里实现文本和图片的负样本配对，image_embeds_neg是image_embeds_neg对应的负样本，text_ids_neg是image_embeds对应的负样本
        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)
        # output_neg是正样本与难负样本融合后的向量
        output_neg = self.text_encoder(text_ids_all,
                                       attention_mask=text_atts_all,
                                       encoder_hidden_states=image_embeds_all,
                                       encoder_attention_mask=image_atts_all,
                                       return_dict=True,
                                       )
        # 计算正样本与正相关、负相关样本融合后的相似度
        vl_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]], dim=0)
        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        irtr_loss = loss_itm + loss_ita
        irtr_loss_ = getattr(self, f"{phase}_irtr_loss")(irtr_loss)
        self.log(f"irtr/{phase}/ita_loss", loss_ita)
        self.log(f"irtr/{phase}/itm_loss", loss_itm)
        self.log(f"irtr/{phase}/irtr_loss", irtr_loss)

        return irtr_loss

    def on_train_epoch_start(self) -> None:
        config = self.hparams.config
        cosine_lr_schedule(self.trainer.optimizers[0], self.current_epoch, config['max_epoch'], config['init_lr'], config['min_lr'])

    def training_step(self, batch, batch_idx):
        model_utils.set_tasks(self)
        if self.trainer.current_epoch > 0:
            alpha = self.hparams.config['alpha']
        else:
            alpha = self.hparams.config['alpha'] * min(1, batch_idx / len(self.trainer.datamodule.train_dataloader()))
        self.hparams.config['cur_alpha'] = alpha

        irtr_loss = self(batch, phase='train')
        return irtr_loss

    def training_epoch_end(self, outs):
        model_utils.epoch_wrapup(self, phase='train')

    def validation_step(self, batch, batch_idx):
        pass
        # model_utils.set_tasks(self)
        # # 如果只有检索任务，则跳过
        # if len(self.current_tasks) == 1 and self.hparams.config['loss_name'].get('irtr', 0) > 0:
        #     return None
        # output = self(batch, phase='val')
        # return output

    def validation_epoch_end(self, outs):
        model_utils.epoch_wrapup(self, phase='val')

    def test_step(self, batch, batch_idx):
        pass
        # model_utils.set_tasks(self)
        # # 如果只有检索任务，则跳过
        # if len(self.current_tasks) == 1 and self.hparams.config['loss_name'].get('irtr', 0) > 0:
        #     return None
        # output = self(batch, phase='test')
        # return output

    def test_epoch_end(self, outs):
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


def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

@torch.no_grad()
def concat_all_gather(tensor, world_size):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if world_size > 1:
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output
    else:
        return tensor


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors, world_size):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    # world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)


'''
    def load_weights(self):
        # 也可以在trainer中添加ckpt： trainer.test(model, datamodule=dm, ckpt_path=str(ckpt))
        ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
        state_dict = ckpt["state_dict"]
        if not self.hparams.config['test_only']:
            state_dict = adapt_position_encoding(state_dict, after=self.hparams.config['image_size'],
                                                 patch_size=self.hparams.config['patch_size'])
        self.load_state_dict(state_dict, strict=False)
        print(f'Load model weights from:{self.hparams.config["load_path"]}')
        del state_dict

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
'''
