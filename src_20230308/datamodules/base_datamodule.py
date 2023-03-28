from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    AutoTokenizer,
    BertTokenizer,
    RobertaTokenizer,
)
from typing import Union, Optional, List, Dict

# 获取分词器，考虑分布式情况
def get_pretrained_tokenizer(from_pretrained):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            return AutoTokenizer.from_pretrained(
                from_pretrained,
                do_lower_case='uncased' in from_pretrained,
                use_fast=False)
        torch.distributed.barrier()
    else:
        return AutoTokenizer.from_pretrained(
            from_pretrained,
            do_lower_case='uncased' in from_pretrained,
            use_fast=False)


class BaseDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.dist = config['dist']
        self.data_dir = config['data_root']
        self.num_workers = config['num_workers']
        self.batch_size = config['per_gpu_batchsize']
        self.eval_batch_size = self.batch_size * 4
        self.shuffle = config['shuffle']
        self.pin_memory = config['pin_memory']

        self.image_size = config['image_size']
        self.max_text_len = config['max_text_len']

        self.image_only = config['image_only']
        self.train_dataset_len = config['train_dataset_len']
        self.val_dataset_len = config['val_dataset_len']
        self.test_dataset_len = config['test_dataset_len']

        # 图片转换器，用于在dataset中将原始图片转换到到image_size大小
        self.train_transform_keys = (
            ['default_train']
            if len(config['train_transform_keys']) == 0
            else config['train_transform_keys']
        )
        self.val_transform_keys = (
            ['default_val']
            if len(config['val_transform_keys']) == 0
            else config['val_transform_keys']
        )

        # 加载分词器
        self.tokenizer = get_pretrained_tokenizer(config['tokenizer'])
        self.vocab_size = self.tokenizer.vocab_size

        # Dataloader 中的数据整理方式
        collator = (
            DataCollatorForWholeWordMask
            if config['whole_word_masking']
            else DataCollatorForLanguageModeling
        )
        self.mlm_collator = collator(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=config['mlm_prob']
        )

        self.setup_flag = False

    # 需要被重写
    @property
    def train_dataset_cls(self):
        raise NotImplementedError("return tuple of train dataset class")

    @property
    def val_dataset_cls(self):
        raise NotImplementedError("return tuple of validation dataset class")

    @property
    def test_dataset_cls(self):
        raise NotImplementedError("return tuple of test dataset class")

    @property
    def dataset_name(self):
        raise NotImplementedError("return name of dataset")

    def set_train_dataset(self):
        self.train_dataset = self.train_dataset_cls(
            self.data_dir,
            self.train_transform_keys,
            split='train',
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            # draw_false_image=self.draw_false_image,
            # draw_false_text=self.draw_false_text,
            image_only=self.image_only,
            tokenizer=self.tokenizer,
            dataset_len=self.train_dataset_len,
        )

    def set_val_dataset(self):
        self.val_dataset = self.val_dataset_cls(
            self.data_dir,
            self.val_transform_keys,
            split='val',
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            # draw_false_image=self.draw_false_image,
            # draw_false_text=self.draw_false_text,
            image_only=self.image_only,
            tokenizer=self.tokenizer,
            dataset_len=self.val_dataset_len,
        )

    def set_test_dataset(self):
        self.test_dataset = self.test_dataset_cls(
            self.data_dir,
            self.val_transform_keys,
            split="test",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            # draw_false_image=self.draw_false_image,
            # draw_false_text=self.draw_false_text,
            image_only=self.image_only,
            tokenizer=self.tokenizer,
            dataset_len=self.test_dataset_len,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.setup_flag:
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()
            # 设置采样器
            if self.dist:
                self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            else:
                self.train_sampler = None
            self.val_sampler = None
            self.test_sampler = None
            # # 设置采样器
            # if self.dist:
            #     self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            #     self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
            #     self.test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
            # else:
            #     self.train_sampler = None
            #     self.val_sampler = None
            #     self.test_sampler = None
        self.setup_flag = True

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            shuffle=self.shuffle if self.train_sampler == None else False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.train_dataset.collate
        )
        return loader

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            sampler=self.val_sampler,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.val_dataset.collate,
        )
        return loader

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            sampler=self.test_sampler,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.test_dataset.collate,
        )
        return loader
