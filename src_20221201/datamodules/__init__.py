
# from .f30k_caption_karpathy_datamodule import F30KCaptionKarpathyDataModule
from .coco_karpathy_datamodule import CocoKarpathyDataModule


_datamodules = {
    # "f30k": F30KCaptionKarpathyDataModule,
    "coco": CocoKarpathyDataModule,
}