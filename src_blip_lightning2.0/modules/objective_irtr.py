'''
检索任务
'''
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import torch.distributed as dist

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
        break

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
        break

    # if dist.distributed_available():
    #     pl_module.trainer.strategy.barrier()

    if dist.is_available():
        torch.distributed.barrier()
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, index_mapper):
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


