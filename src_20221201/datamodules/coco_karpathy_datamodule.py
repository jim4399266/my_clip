import sys
sys.path.append('..')

from datasets.coco_karpathy_dataset import CocoKarpathyDataset
# from .base_datamodule import BaseDataModule
from .base_datamodule import BaseDataModule

class CocoKarpathyDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # 初始化数据集
    @property
    def dataset_cls(self):
        return CocoKarpathyDataset

    # @property
    # def dataset_cls_no_false(self):
    #     return CocoKarpathyDataset

    @property
    def dataset_name(self):
        return 'coco'
