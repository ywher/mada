"""Generic selected-image list dataset for MADAv2 stage 1."""

from .list_common import ListSegmentationDataset


class List_active_loader(ListSegmentationDataset):
    def __init__(self, cfg, writer, logger, augmentations=None, **kwargs):
        super().__init__(
            cfg,
            writer,
            logger,
            augmentations=augmentations,
            active=True,
            **kwargs,
        )
