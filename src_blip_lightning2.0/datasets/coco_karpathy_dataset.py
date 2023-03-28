# from .base_dataset import BaseDataset
import io
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
from typing import Union, Optional, List, Dict
import sys
sys.path.append('..')


from .base_dataset import CocoKarpathyBaseDataset
import io
from PIL import Image
import torch
from torch.utils.data import Dataset

class CocoKarpathyDataset(CocoKarpathyBaseDataset):
    def __init__(self, *args, split='', names='', **kwargs):
        assert split in ['train', 'val', 'test']
        self.split = split
        if split == "train":
            names = ["coco_caption_karpathy_train", "coco_caption_karpathy_restval"]
        elif split == "val":
            names = ["coco_caption_karpathy_val"]
        elif split == "test":
            names = ["coco_caption_karpathy_test"]
        super().__init__(*args, **kwargs, names=names, text_column_name='caption')
        print(f'CocoCaptionKarpathyDataset {split} len : {len(self)}')


class CocoKarpathyRecallDataset(CocoKarpathyBaseDataset):
    def __init__(self, *args, split='', names='', **kwargs):
        assert split in ['train', 'val', 'test']
        self.split = split
        if split == "train":
            names = ["coco_caption_karpathy_train", "coco_caption_karpathy_restval"]
        elif split == "val":
            names = ["coco_caption_karpathy_val"]
        elif split == "test":
            names = ["coco_caption_karpathy_test"]
        super().__init__(*args, **kwargs, names=names, text_column_name='caption')
        print(f'CocoCaptionKarpathyDataset {split} len : {len(self)}')

    def __getitem__(self, index):
        # 在测试时，需要返回所有图片和文本的信息
        ret = dict()
        try:
            ret.update(self.get_image(index))
            if not self.image_only:
                # 测试时，返回一张图片对应的一组文本
                text = self.get_text(index)
                ret.update(text)
        except :
            print(f"Error while read file idx {index} in {self.names[0]}")
        # if 'test' in self.split:
        #     index_, question_index_ = self.index_mapper[index]
        #     iid = self.table['image_id'][index_].as_py()
        #     iid = int(iid.split('.')[0].split('_')[-1])
        #     ret.update({'iid': iid})
        return ret

    def get_text(self, image_index, text_key='caption'):
        # 测试时，返回一张图片对应的一组文本
        texts = self.all_texts[image_index]
        encodings = self.tokenizer(
            texts,  # 这里是一个列表，包含每张图片对应的一组文本
            padding='max_length',
            max_length=self.max_text_len,
            truncation=True,
            return_special_tokens_mask=True,  # 遮住特殊token的mask
            return_tensors='pt'
        )
        # 注意区分key中的text和cap，在collate会有不同处理
        return {
            'text': None,  # 正例的原文
            'text_encodings': encodings,  # 正例的encoding
            'text_index': None,  # 正例的下标
            'text_list': texts,  # 图片对应的文本列表
            'text_list_index': [image_index] * len(texts)
        }


