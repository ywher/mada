"""Generic unlabeled-target list dataset for MADAv2 stage 2."""

from .list_common import ListSegmentationDataset


class List_AEL_loader(ListSegmentationDataset):
    def __init__(self, cfg, writer, logger, augmentations=None, **kwargs):
        super().__init__(
            cfg,
            writer,
            logger,
            augmentations=augmentations,
            ael=True,
            **kwargs,
        )
