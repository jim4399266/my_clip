'''
检索任务
'''
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import torch.distributed as dist
from .dist_utils import all_gather_with_grad, concat_all_gather

def in_modality_g2l_loss(local_feature, global_feature, temp=1., attention_mask=None):
    '''
    :param global_feature: bs, dim
    :param local_feature: bs, len, dim
    :param temp: matmul temperature
    :param attention_mask: text attention mask: bs, len
    :return:
    '''
    FILL = float('-inf')
    global_feature = global_feature.unsqueeze(1)   # bs, 1, dim
    bs, local_len, dim = local_feature.size()
    # 正样本对应的 global_feature 和 local_feature 进行点积，越小越好
    logits_pos = torch.matmul(local_feature, global_feature.permute(0, 2, 1)) / temp # bs, len, 1
    # 对文本填充的部分进行遮蔽，遮蔽部分赋值 负无穷，去除 softmax 时的影响
    if attention_mask is not None:
        tmp_mask = attention_mask.unsqueeze(-1)
        logits_pos = logits_pos.masked_fill(tmp_mask != 1, FILL)

    # 接下来对所有负样本进行点积，越大越好
    # 每个样本的 global feature 需要与所有样本的 local feature 计算
    # 相当于每个正样本中的 global feature:[1, dim] 需要与 每个负样本中的每个token(bs * len)进行点积
    # 即  global_feature_n: bs, dim        local_feature_n: (bs * local_len), dim
    global_feature_n, local_feature_n = global_feature.reshape(-1, dim), local_feature.reshape(-1, dim)
    logits_neg = torch.matmul(global_feature_n, local_feature_n.T) / temp  # bs, (bs * local_len)
    logits_neg = logits_neg.reshape(bs, bs, local_len)  # bs, bs, local_len
    # 首先需要对正样本进行遮蔽
    tmp_mask = 1 - torch.eye(bs)[:, :, None].to(logits_neg.device)
    logits_neg = logits_neg.masked_fill(tmp_mask != 1, FILL)
    # 对文本填充的部分进行遮蔽，遮蔽部分赋值 负无穷，去除 softmax 时的影响
    if attention_mask is not None:
        tmp_mask = attention_mask.unsqueeze(0)   # 1, bs, len
        logits_neg = logits_neg.masked_fill(tmp_mask != 1, FILL)

    # 每个正样本的得分需要与所有其他负样本得分进行 softmax 计算，要使 正样本的得分 占比更大
    # 正样本有 local_len 个，负样本有 (bs * local_len) 个
    # 因此需要将负样本进行维度变换:
    logits_neg = logits_neg.reshape(bs, -1).unsqueeze(1).expand(-1, local_len, -1) # bs, local_len, (bs * local_len)

    # Since this is multiclass, we concat the positive along the class dimension before performing log softmax.
    pred_lgt = torch.cat([logits_pos, logits_neg], dim=-1)
    pred_log = F.log_softmax(pred_lgt, dim=-1)

    # The positive score is the first element of the log softmax
    if attention_mask is not None:
        pred_log = -pred_log[:, :, 0].squeeze()
        pred_log = pred_log.masked_fill(attention_mask != 1, 0.)
        loss = (torch.sum(pred_log, dim=1) / torch.sum(attention_mask, dim=1)).mean()
    else:
        pred_log = -pred_log[:, :, 0]
        loss = pred_log.mean()
    return loss



# def in_modality_g2l_loss(local_feature, global_feature, temp, attention_mask=None):
#     '''
#     :param global_feature: bs, dim
#     :param local_feature: bs, len, dim
#     :param temp: matmul temperature
#     :param attention_mask: text attention mask: bs, len
#     :return:
#     '''
#     FILL = float('-inf')
#     global_feature = global_feature.unsqueeze(1)   # bs, 1, dim
#     bs, local_len, dim = local_feature.size()
#
#     # 每个样本的 global feature 需要与所有样本的 local feature 计算
#     # 相当于每个正样本中的 global feature:[1, dim] 需要与 每个负样本中的每个token(bs * len)进行点积
#     # 即  global_feature_n: bs, dim        local_feature_n: (bs * local_len), dim
#     global_feature_n, local_feature_n = global_feature.reshape(-1, dim), local_feature.reshape(-1, dim)
#     logits = torch.matmul(global_feature_n, local_feature_n.T)  / temp # bs, (bs * local_len)
#     logits = logits.reshape(bs, bs, local_len)  # bs, bs, local_len # 其中对角线上的是正样本，其余为负样本
#     # 对于文本，需要对填充部分进行遮蔽，遮蔽部分赋值 负无穷，去除 softmax 时的影响
#     if attention_mask is not None:
#         tmp_mask = attention_mask.unsqueeze(0)   # 1, bs, len，会自动广播
#         logits = logits.masked_fill(tmp_mask != 1, FILL)
#     pred_log = F.log_softmax(logits, dim=1)   # 这里 dim=0 或 1 都可以
#     labels = torch.eye(bs).unsqueeze(-1).to(attention_mask.device)  # bs, bs, 1
#     if attention_mask is not None:
#         tmp_mask = attention_mask.unsqueeze(0).expand(bs, -1, -1) # bs, bs, len
#         labels = labels.masked_fill(tmp_mask != 1, -100)
#
#     loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1) * sim_targets, dim=1).mean()
#     loss = (torch.sum(-pred_log[:, :, 0].squeeze(), dim=1) / torch.sum(attention_mask, dim=1)).mean()
#
#
#     # 可以去掉这部分，cross_entropy 会忽略
#
#
#     # 构建正样本标签，文本填充标记-100
#     # labels = 1 - torch.eye(bs).unsqueeze(-1).to(attention_mask.device)  # bs, bs, 1
#     # if attention_mask is not None:
#     #     tmp_mask = attention_mask.unsqueeze(0).expand(bs, -1, -1) # bs, bs, len
#     #     labels = labels.masked_fill(tmp_mask != 1, -100)
#     #
#     # # 使用 F.cross_entropy 计算损失，因为可以用到 ignore_index
#     # # F.cross_entropy 要求 pred: [N, C, d1, d2, ..., dk]    label: [N, d1, d2,...,dk]
#     # # 即 pred 比 label 多出一个 class 的维度，因此需要将 logits 扩充最后一个维度实现 token 的二分类
#     # logits = torch.sigmoid(logits).unsqueeze(-1)  # bs, bs, local_len, 1
#     # logits = torch.cat([logits, 1-logits], dim=-1) # bs, bs, local_len, 2
#     # loss = F.cross_entropy(logits.permute(0, 3, 1, 2), labels, ignore_index=-100)
#     # return loss


def in_modality_g2l_loss_o(local_feature, global_feature, temp, attention_mask=None):
    fill = float('-inf')
    global_feature = global_feature.unsqueeze(1)
    bs, local_len, dim = local_feature.size()

    # Inner product for positive samples. Outer product for negative. We need to do it this way
    # for the multiclass loss. For the outer product, we want a 'bs, bs, local_len, 1' tensor.
    # 对应正样本的 global_feature 和 local_feature 进行点积
    u_pos = torch.matmul(local_feature, global_feature.permute(0, 2, 1)).unsqueeze(-1) / temp # bs, local_len, 1, 1
    # if 1 comes frm text, then attention_mask is not None
    if attention_mask is not None:
        temp_mask = attention_mask.unsqueeze(-1).unsqueeze(-1)
        u_pos = u_pos.masked_fill(temp_mask != 1, fill)

    # 为了让正样本和所有负样本的所有 local_feature 进行计算，需要将 local_feature 前两个维度合并，留最后一个维度进行计算
    local_feature_n = local_feature.reshape(-1, dim)  # (bs * local_len) * dim
    global_feature_n = global_feature.reshape(-1, dim)  # bs * dim
    u_neg = torch.matmul(global_feature_n, local_feature_n.T) / temp  # bs, (bs * local_len)
    u_neg = u_neg.reshape(bs, 1, bs, local_len).permute(0, 2, 3, 1)    # bs, bs, local_len, 1

    # We need to mask the diagonal part of the negative tensor
    # 需要将正样本遮住
    mask = torch.eye(bs)[:, :, None, None].to(local_feature.device)  # bs, bs, 1, 1
    n_mask = 1 - mask

    # Masking is done by shifting the diagonal before exp
    u_neg = u_neg.masked_fill(n_mask != 1, fill)   # mask out "self" examples

    # if 1 comes from text, we mask out the padding tokens
    if attention_mask is not None:
        temp_mask = attention_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)  # bs, bs, local_len, 1
        # u_n = (temp_mask * u_n) - (fill * (1 - temp_mask))
        u_neg = u_neg.masked_fill(temp_mask != 1, fill)
    u_neg = u_neg.reshape(bs, bs * local_len, 1).\
        unsqueeze(1).expand(-1, local_len, -1, -1) # bs, local_len, (bs*local_len), 1

    # Since this is multiclass, we concat the positive along the class dimension before performing log softmax.
    pred_lgt = torch.cat([u_pos, u_neg], dim=2)
    pred_log = F.log_softmax(pred_lgt, dim=2)
    # pred_log = pred_log.masked_fill(attention_mask != 1, 0.)

    # The positive score is the first element of the log softmax
    if attention_mask is not None:
        loss = (torch.sum(-pred_log[:, :, 0].squeeze(), dim=1) / torch.sum(attention_mask, dim=1)).mean()
    else:
        loss = -pred_log[:, :, 0].mean()
    return loss


def train_irtr(pl_module, batch, phase):
    # TCL方法，三重损失
    with torch.no_grad():
        pl_module.temp.clamp_(0.001, 0.5)
    image = batch['image']
    caption = batch['text']
    alpha = pl_module.hparams.config['cur_alpha']
    idx = batch['image_index']

    image_embeds = pl_module.visual_encoder(image)
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
    image_feat = F.normalize(pl_module.vision_proj(image_embeds[:, 0, :]), dim=-1)

    text = pl_module.tokenizer(caption, padding='max_length', truncation=True, max_length=35,
                          return_tensors="pt").to(image.device)

    text_output = pl_module.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                    return_dict=True, mode='text')
    text_feat = F.normalize(pl_module.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

    ###============== Image-text Contrastive Learning ===================###
    idx = idx.view(-1, 1)
    idx_all = torch.cat([idx.t(), pl_module.idx_queue.clone().detach()], dim=1)
    pos_idx = torch.eq(idx, idx_all).float()
    sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

    # get momentum features
    with torch.no_grad():
        pl_module._momentum_update()
        # -------  all image momentum features  -----------
        image_embeds_m = pl_module.visual_encoder_m(image)
        image_feat_m = F.normalize(pl_module.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
        image_feat_m_all = torch.cat([image_feat_m.t(), pl_module.image_queue.clone().detach()], dim=1)
        #  momentum image local features
        image_feat_m_l = F.normalize(pl_module.vision_proj_m(image_embeds_m[:, 1:, :]), dim=-1)
        image_feat_m_l = pl_module.patch_pooling(image_feat_m_l)  # pooling for image patches

        # -------  all text momentum features  -----------
        text_output_m = pl_module.text_encoder_m(text.input_ids, attention_mask=text.attention_mask,
                                            return_dict=True, mode='text')
        text_feat_m = F.normalize(pl_module.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
        text_feat_m_all = torch.cat([text_feat_m.t(), pl_module.text_queue.clone().detach()], dim=1)
        # momentum text local features
        text_feat_m_l = F.normalize(pl_module.text_proj_m(text_output_m.last_hidden_state[:, 1:, :]), dim=-1)

        if pl_module.distill:
            sim_i2t_m = image_feat_m @ text_feat_m_all / pl_module.temp
            sim_t2i_m = text_feat_m @ image_feat_m_all / pl_module.temp

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

    sim_i2t = image_feat @ text_feat_m_all / pl_module.temp
    sim_t2i = text_feat @ image_feat_m_all / pl_module.temp

    if pl_module.distill:
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
    else:
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1).mean()

    # add in-modality g2l loss (in-modality global to local)
    loss_t2t_IM_g2l = in_modality_g2l_loss(text_feat_m_l, text_feat, pl_module.temp, text.attention_mask[:, 1:])
    loss_i2i_IM_g2l = in_modality_g2l_loss(image_feat_m_l, image_feat, pl_module.temp)

    # add in-modality g2g loss (in-modality local to local)
    sim_i2i = image_feat @ image_feat_m_all / self.temp
    sim_t2t = text_feat @ text_feat_m_all / self.temp
    loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1) * sim_targets, dim=1).mean()
    loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1) * sim_targets, dim=1).mean()

    loss_itc = (loss_i2t + loss_t2i + loss_i2i_IM_g2l + loss_t2t_IM_g2l + loss_t2t + loss_i2i) / 6

    idxs = concat_all_gather(idx, world_size=pl_module.trainer.world_size)
    pl_module._dequeue_and_enqueue(image_feat_m, text_feat_m, idxs)

    ###============== Image-text Matching ===================###
    encoder_input_ids = text.input_ids.clone()
    encoder_input_ids[:, 0] = pl_module.tokenizer.enc_token_id

    # forward the positve image-text pair         # 正相关的图文对进行融合，得到output_pos，是正相关图文对融合后的向量
    bs = image.size(0)
    output_pos = pl_module.text_encoder(encoder_input_ids,
                                   attention_mask=text.attention_mask,
                                   encoder_hidden_states=image_embeds,
                                   encoder_attention_mask=image_atts,
                                   return_dict=True,
                                   )

    if pl_module.negative_all_rank:  # 如果是分布式，从所有卡中抽取负样本
        # compute sample similarity
        with torch.no_grad():
            mask = torch.eq(idx, idxs.t())

            image_feat_world = concat_all_gather(image_feat, pl_module.trainer.world_size)
            text_feat_world = concat_all_gather(text_feat, pl_module.trainer.world_size)

            sim_i2t = image_feat @ text_feat_world.t() / pl_module.temp
            sim_t2i = text_feat @ image_feat_world.t() / pl_module.temp

            weights_i2t = F.softmax(sim_i2t, dim=1)
            weights_i2t.masked_fill_(mask, 0)

            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_t2i.masked_fill_(mask, 0)

        image_embeds_world = all_gather_with_grad(image_embeds, pl_module.trainer.world_size)

        # select a negative image (from all ranks) for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text (from all ranks) for each image
        input_ids_world = concat_all_gather(encoder_input_ids, pl_module.trainer.world_size)
        att_mask_world = concat_all_gather(text.attention_mask, pl_module.trainer.world_size)

        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(input_ids_world[neg_idx])
            text_atts_neg.append(att_mask_world[neg_idx])

    else:  # 仅从当前卡上的批次中抽取负样本
        with torch.no_grad():
            mask = torch.eq(idx, idx.t())

            sim_i2t = image_feat @ text_feat.t() / pl_module.temp
            sim_t2i = text_feat @ image_feat.t() / pl_module.temp

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
    output_neg = pl_module.text_encoder(text_ids_all,
                                   attention_mask=text_atts_all,
                                   encoder_hidden_states=image_embeds_all,
                                   encoder_attention_mask=image_atts_all,
                                   return_dict=True,
                                   )
    # 计算正样本与正相关、负相关样本融合后的相似度
    vl_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]], dim=0)
    vl_output = pl_module.itm_head(vl_embeddings)

    itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                           dim=0).to(image.device)
    loss_itm = F.cross_entropy(vl_output, itm_labels)

    irtr_loss = loss_itm + loss_itc
    irtr_loss_ = getattr(pl_module, f"{phase}_irtr_loss")(irtr_loss)
    pl_module.log(f"irtr/{phase}/itc_loss", loss_itc)
    pl_module.log(f"irtr/{phase}/itm_loss", loss_itm)
    pl_module.log(f"irtr/{phase}/irtr_loss", irtr_loss)
    return irtr_loss

# def train_irtr(pl_module, batch, phase):
#     with torch.no_grad():
#         pl_module.temp.clamp_(0.001, 0.5)
#     image = batch['image']
#     caption = batch['text']
#     alpha = pl_module.hparams.config['cur_alpha']
#     idx = batch['image_index']
#
#     image_embeds = pl_module.visual_encoder(image)
#     image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
#     image_feat = F.normalize(pl_module.vision_proj(image_embeds[:, 0, :]), dim=-1)
#
#     text = pl_module.tokenizer(caption, padding='max_length', truncation=True, max_length=35,
#                           return_tensors="pt").to(image.device)
#
#     text_output = pl_module.text_encoder(text.input_ids, attention_mask=text.attention_mask,
#                                     return_dict=True, mode='text')
#     text_feat = F.normalize(pl_module.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)
#
#     ###============== Image-text Contrastive Learning ===================###
#     idx = idx.view(-1, 1)
#     idx_all = torch.cat([idx.t(), pl_module.idx_queue.clone().detach()], dim=1)
#     pos_idx = torch.eq(idx, idx_all).float()
#     sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
#
#     # get momentum features
#     with torch.no_grad():
#         pl_module._momentum_update()
#         image_embeds_m = pl_module.visual_encoder_m(image)
#         image_feat_m = F.normalize(pl_module.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
#         image_feat_m_all = torch.cat([image_feat_m.t(), pl_module.image_queue.clone().detach()], dim=1)
#
#         text_output_m = pl_module.text_encoder_m(text.input_ids, attention_mask=text.attention_mask,
#                                             return_dict=True, mode='text')
#         text_feat_m = F.normalize(pl_module.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
#         text_feat_m_all = torch.cat([text_feat_m.t(), pl_module.text_queue.clone().detach()], dim=1)
#
#         sim_i2t_m = image_feat_m @ text_feat_m_all / pl_module.temp
#         sim_t2i_m = text_feat_m @ image_feat_m_all / pl_module.temp
#
#         sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
#         sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
#
#     sim_i2t = image_feat @ text_feat_m_all / pl_module.temp
#     sim_t2i = text_feat @ image_feat_m_all / pl_module.temp
#
#     loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
#     loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
#
#     loss_itc = (loss_i2t + loss_t2i) / 2
#
#     idxs = concat_all_gather(idx, world_size=pl_module.trainer.world_size)
#     pl_module._dequeue_and_enqueue(image_feat_m, text_feat_m, idxs)
#
#     ###============== Image-text Matching ===================###
#     encoder_input_ids = text.input_ids.clone()
#     encoder_input_ids[:, 0] = pl_module.tokenizer.enc_token_id
#
#     # forward the positve image-text pair         # 正相关的图文对进行融合，得到output_pos，是正相关图文对融合后的向量
#     bs = image.size(0)
#     output_pos = pl_module.text_encoder(encoder_input_ids,
#                                    attention_mask=text.attention_mask,
#                                    encoder_hidden_states=image_embeds,
#                                    encoder_attention_mask=image_atts,
#                                    return_dict=True,
#                                    )
#
#     if pl_module.negative_all_rank:  # 如果是分布式，从所有卡中抽取负样本
#         # compute sample similarity
#         with torch.no_grad():
#             mask = torch.eq(idx, idxs.t())
#
#             image_feat_world = concat_all_gather(image_feat, pl_module.trainer.world_size)
#             text_feat_world = concat_all_gather(text_feat, pl_module.trainer.world_size)
#
#             sim_i2t = image_feat @ text_feat_world.t() / pl_module.temp
#             sim_t2i = text_feat @ image_feat_world.t() / pl_module.temp
#
#             weights_i2t = F.softmax(sim_i2t, dim=1)
#             weights_i2t.masked_fill_(mask, 0)
#
#             weights_t2i = F.softmax(sim_t2i, dim=1)
#             weights_t2i.masked_fill_(mask, 0)
#
#         image_embeds_world = all_gather_with_grad(image_embeds, pl_module.trainer.world_size)
#
#         # select a negative image (from all ranks) for each text
#         image_embeds_neg = []
#         for b in range(bs):
#             neg_idx = torch.multinomial(weights_t2i[b], 1).item()
#             image_embeds_neg.append(image_embeds_world[neg_idx])
#         image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
#
#         # select a negative text (from all ranks) for each image
#         input_ids_world = concat_all_gather(encoder_input_ids, pl_module.trainer.world_size)
#         att_mask_world = concat_all_gather(text.attention_mask, pl_module.trainer.world_size)
#
#         text_ids_neg = []
#         text_atts_neg = []
#         for b in range(bs):
#             neg_idx = torch.multinomial(weights_i2t[b], 1).item()
#             text_ids_neg.append(input_ids_world[neg_idx])
#             text_atts_neg.append(att_mask_world[neg_idx])
#
#     else:  # 仅从当前卡上的批次中抽取负样本
#         with torch.no_grad():
#             mask = torch.eq(idx, idx.t())
#
#             sim_i2t = image_feat @ text_feat.t() / pl_module.temp
#             sim_t2i = text_feat @ image_feat.t() / pl_module.temp
#
#             weights_i2t = F.softmax(sim_i2t, dim=1)
#             weights_i2t.masked_fill_(mask, 0)
#
#             weights_t2i = F.softmax(sim_t2i, dim=1)
#             weights_t2i.masked_fill_(mask, 0)
#
#             # select a negative image (from same rank) for each text
#         image_embeds_neg = []
#         for b in range(bs):
#             neg_idx = torch.multinomial(weights_t2i[b], 1).item()
#             image_embeds_neg.append(image_embeds[neg_idx])
#         image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
#
#         # select a negative text (from same rank) for each image
#         text_ids_neg = []
#         text_atts_neg = []
#         for b in range(bs):
#             neg_idx = torch.multinomial(weights_i2t[b], 1).item()
#             text_ids_neg.append(encoder_input_ids[neg_idx])
#             text_atts_neg.append(text.attention_mask[neg_idx])
#
#     text_ids_neg = torch.stack(text_ids_neg, dim=0)
#     text_atts_neg = torch.stack(text_atts_neg, dim=0)
#     # 这里实现文本和图片的负样本配对，image_embeds_neg是image_embeds_neg对应的负样本，text_ids_neg是image_embeds对应的负样本
#     text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)
#     text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)
#
#     image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
#     image_atts_all = torch.cat([image_atts, image_atts], dim=0)
#     # output_neg是正样本与难负样本融合后的向量
#     output_neg = pl_module.text_encoder(text_ids_all,
#                                    attention_mask=text_atts_all,
#                                    encoder_hidden_states=image_embeds_all,
#                                    encoder_attention_mask=image_atts_all,
#                                    return_dict=True,
#                                    )
#     # 计算正样本与正相关、负相关样本融合后的相似度
#     vl_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]], dim=0)
#     vl_output = pl_module.itm_head(vl_embeddings)
#
#     itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
#                            dim=0).to(image.device)
#     loss_itm = F.cross_entropy(vl_output, itm_labels)
#
#     irtr_loss = loss_itm + loss_itc
#     irtr_loss_ = getattr(pl_module, f"{phase}_irtr_loss")(irtr_loss)
#     pl_module.log(f"irtr/{phase}/itc_loss", loss_itc)
#     pl_module.log(f"irtr/{phase}/itm_loss", loss_itm)
#     pl_module.log(f"irtr/{phase}/irtr_loss", irtr_loss)
#     return irtr_loss


@torch.no_grad()
def val_irtr(pl_module, data_loader):
    device = pl_module.device
    config = pl_module.hparams.config

    # texts = data_loader.dataset.text
    texts = data_loader.dataset.corpus
    num_text = len(texts)
    # num_text = 1024
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    # print('Computing text features...')
    for i in tqdm(range(0, num_text, text_bs), desc='Computing text features...'):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = pl_module.tokenizer(text, padding='max_length', truncation=True, max_length=35,
                                     return_tensors="pt").to(device)
        text_output = pl_module.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        text_embed = F.normalize(pl_module.text_proj(text_output.last_hidden_state[:, 0, :]))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    text_ids[:, 0] = pl_module.tokenizer.enc_token_id

    image_feats = []
    image_embeds = []
    for batch in tqdm(data_loader, desc='Computing image features...'):
        image, img_id = batch['image'], batch['image_index']
        image = image.to(device)
        image_feat = pl_module.visual_encoder(image)
        image_embed = pl_module.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)

        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)

    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full((len(data_loader.dataset.all_texts), len(texts)), -100.0).to(device)

    num_devices = pl_module.trainer.world_size
    rank = pl_module.trainer.global_rank
    step = sims_matrix.size(0) // num_devices + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in tqdm(enumerate(sims_matrix[start:end]), desc='image2text recalling'):
        topk_sim, topk_idx = sims.topk(k=config['top_k'], dim=0)

        encoder_output = image_feats[start + i].repeat(config['top_k'], 1, 1).to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = pl_module.text_encoder(text_ids[topk_idx],
                                    attention_mask=text_atts[topk_idx],
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    )
        score = pl_module.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_i2t[start + i, topk_idx] = score + topk_sim

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.all_texts)), -100.0).to(device)

    num_devices = pl_module.trainer.world_size
    rank = pl_module.trainer.global_rank
    step = sims_matrix.size(0) // num_devices + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in tqdm(enumerate(sims_matrix[start:end]), desc='text2image recalling'):
        topk_sim, topk_idx = sims.topk(k=config['top_k'], dim=0)
        encoder_output = image_feats[topk_idx.cpu()].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = pl_module.text_encoder(text_ids[start + i].repeat(config['top_k'], 1),
                                    attention_mask=text_atts[start + i].repeat(config['top_k'], 1),
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    )
        score = pl_module.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim

    # if dist.distributed_available():
    #     pl_module.trainer.strategy.barrier()

    if dist.is_available():
        torch.distributed.barrier()
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


@torch.no_grad()
def recall_eval(scores_i2t, scores_t2i, index_mapper):
    img2txt, txt2img = {}, {}
    for k, v in index_mapper.items():
        # t_idx = k         i_idx = v[0]
        txt2img[k] = v[0]
        img2txt.setdefault(v[0], []).append(k)

    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                   'r_mean': r_mean}
    return eval_result


