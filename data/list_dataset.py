"""Generic list dataset for source, target, and validation data."""

from .list_common import ListSegmentationDataset


class List_loader(ListSegmentationDataset):
    def __init__(self, cfg, writer, logger, augmentations=None, **kwargs):
        super().__init__(
            cfg, writer, logger, augmentations=augmentations, **kwargs
        )
