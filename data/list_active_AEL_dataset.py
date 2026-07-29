"""Generic selected-image dataset with optional AEL copy-paste."""

from .list_common import ListSegmentationDataset


class List_active_AEL_loader(ListSegmentationDataset):
    def __init__(
        self,
        cfg,
        writer,
        logger,
        augmentations=None,
        acp=False,
        prob=0.5,
        **kwargs,
    ):
        super().__init__(
            cfg,
            writer,
            logger,
            augmentations=augmentations,
            active=True,
            ael=True,
            acp=acp,
            prob=prob,
            **kwargs,
        )
