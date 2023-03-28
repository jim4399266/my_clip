import functools
from typing import Union, List, Dict, Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from . import _datamodules

class MTDataModule(LightningDataModule):
    def __init__(self, config, dist=False):
        '''
        :param config:
        :param dist:  是否分布式训练
        '''
        super().__init__()
        self.dm_keys = config['datasets']
        assert len(self.dm_keys) > 0
        self.dm_dict = {key: _datamodules[key](config) for key in self.dm_keys}  # 利用一个通用类创建各个数据集的datamodule
        self.dm_list = [dm for key, dm in self.dm_dict.items()]

        self.batch_size = config['per_gpu_batchsize']
        self.num_workers = config['num_workers']
        # self.vocab_size = self.dm_list[0].vocab_size
        self.dist = dist
        self.shuffle = config['shuffle']

    def prepare_data(self) -> None:
        for dm in self.dm_list:
            dm.prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        for dm in self.dm_list:
            # 初始化每个数据集的datamodule
            dm.setup()

        self.train_dataset = ConcatDataset([dm.train_dataset for dm in self.dm_list])
        self.val_dataset = ConcatDataset([dm.val_dataset for dm in self.dm_list])
        self.test_dataset = ConcatDataset([dm.test_dataset for dm in self.dm_list])

        # 为DataLoader设置collate
        self.collate = functools.partial(
            self.dm_list[0].train_dataset.collate, mlm_collator=self.dm_list[0].mlm_collator,
        )

        # 设置采样器
        if self.dist:
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
            self.test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
        else:
            self.train_sampler = None
            self.val_sampler = None
            self.test_sampler = None

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            shuffle=self.shuffle if self.train_sampler == None else False,
            collate_fn=self.collate,
        )
        return loader

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate,
        )
        return loader

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate,
        )
        return loader