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

    def __getitem__(self, index):
        suite = self.get_suite(index)
        if 'test' in self.split:
            index_, question_index_ = self.index_mapper[index]
            iid = self.table['image_id'][index_].as_py()
            iid = int(iid.split('.')[0].split('_')[-1])
            suite.update({'iid': iid})
        return suite


class CocoKarpathyRecallDataset(CocoKarpathyBaseDataset):
    def __init__(self, tensor, ids, masks=None):
        super().__init__()
        self.tensor = tensor
        self.ids = ids
        self.masks = masks

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, index):
        t = self.tensor[index]
        i = self.ids[index]
        m = None if self.masks == None else self.masks[index]
        return (t, i, m,)

    def collate(self, batch, mlm_collator=None):
        batch_tensor = torch.stack([item[0] for item in batch], dim=0).float()
        batch_ids = torch.stack([item[1] for item in batch], dim=0).long()
        batch_masks = None if self.masks == None else torch.stack([item[2] for item in batch], dim=0).float()
        return (batch_tensor, batch_ids, batch_masks)
