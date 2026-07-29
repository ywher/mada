"""List-based segmentation datasets used by the reproducible MADAv2 pipeline."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.base_dataset import BaseDataset


IGNORE_INDEX = 255
CITYSCAPES_TO_SYN_CITY = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for _city_id, _syn_id in {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    10: 9,
    11: 10,
    12: 11,
    13: 12,
    15: 13,
    17: 14,
    18: 15,
}.items():
    CITYSCAPES_TO_SYN_CITY[_city_id] = _syn_id
CITYSCAPES_TO_SYN_CITY[IGNORE_INDEX] = IGNORE_INDEX


def _read_pairs(list_path: str | os.PathLike) -> list[tuple[str, str]]:
    pairs = []
    with open(list_path, "r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = (
                [field.strip() for field in line.split(",", 1)]
                if "," in line
                else line.split(maxsplit=1)
            )
            if len(fields) != 2:
                raise ValueError(
                    f"{list_path}:{line_no}: expected image and label paths"
                )
            pairs.append((fields[0], fields[1]))
    if not pairs:
        raise ValueError(f"No image/label pairs found in {list_path}")
    return pairs


class ListSegmentationDataset(BaseDataset):
    """Read image/label pairs from a comma- or whitespace-delimited list."""

    def __init__(
        self,
        cfg,
        writer,
        logger,
        augmentations=None,
        active=False,
        ael=False,
        acp=False,
        prob=0.5,
    ):
        del writer
        super().__init__(cfg)
        self.cfg = cfg
        self.root = Path(cfg["rootpath"]).expanduser().resolve()
        self.list_path = Path(
            cfg.get("active_list_path") if active else cfg["list_path"]
        ).expanduser().resolve()
        self.files = _read_pairs(self.list_path)
        self.augmentations = augmentations
        self.is_transform = cfg.get("is_transform", True)
        self.n_classes = int(cfg.get("n_class", 19))
        self.ignore_index = IGNORE_INDEX
        self.class_set = cfg.get("class_set", "cityscapes")
        self.active = active
        self.ael = ael
        self.acp = bool(acp and active and ael)
        self.acp_probability = float(prob)
        self.resize = cfg.get("resize")
        self.mean = np.asarray(
            cfg.get("mean", [0.0, 0.0, 0.0]), dtype=np.float32
        )
        self.std = np.asarray(
            cfg.get("std", [255.0, 255.0, 255.0]), dtype=np.float32
        )
        if np.any(self.std <= 0):
            raise ValueError("Dataset normalization std must be positive")
        self.return_path = bool(cfg.get("return_path", False))
        logger.info(
            "Loaded %d samples from %s (class_set=%s)",
            len(self.files),
            self.list_path,
            self.class_set,
        )

    def __len__(self):
        return len(self.files)

    def _load_pair(self, index):
        image_rel, label_rel = self.files[index]
        image_path = self.root / image_rel
        label_path = self.root / label_rel
        if not image_path.is_file():
            raise FileNotFoundError(f"Image does not exist: {image_path}")
        if not label_path.is_file():
            raise FileNotFoundError(f"Label does not exist: {label_path}")
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)
        if self.resize:
            height, width = (int(v) for v in self.resize)
            image = image.resize((width, height), Image.BILINEAR)
            label = label.resize((width, height), Image.NEAREST)
        image = np.asarray(image, dtype=np.uint8)
        label = np.asarray(label, dtype=np.uint8)
        if self.class_set == "syn_city":
            label = CITYSCAPES_TO_SYN_CITY[label]
        return image, label, str(image_rel)

    def _transform(self, image, label):
        if self.augmentations is not None:
            image, label = self.augmentations(image, label)
        image = np.asarray(image, dtype=np.float32)
        label = np.asarray(label, dtype=np.int64)
        if self.is_transform:
            image = (image - self.mean) / self.std
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            label = torch.from_numpy(label).long()
        return image, label

    def __getitem__(self, index):
        image, label, image_name = self._load_pair(index)
        image, label = self._transform(image, label)
        sample_id = image_name if self.return_path else index

        if not self.acp:
            return image, label, sample_id

        # Keep a fixed five-item return contract for DataLoader collation while
        # preserving the original probabilistic ACP behavior. An all-zero
        # paste label is treated as a no-op by dynamic_copy_paste.
        if random.random() > self.acp_probability:
            paste_index = random.randrange(len(self.files))
            paste_image, paste_label, _ = self._load_pair(paste_index)
            paste_image, paste_label = self._transform(paste_image, paste_label)
        else:
            paste_image = image.clone()
            paste_label = torch.zeros_like(label)
        return image, label, paste_image, paste_label, sample_id


def build_list_dataset(
    cfg,
    writer,
    logger,
    augmentations=None,
    *,
    active=False,
    ael=False,
    acp=False,
    prob=0.5,
):
    return ListSegmentationDataset(
        cfg,
        writer,
        logger,
        augmentations=augmentations,
        active=active,
        ael=ael,
        acp=acp,
        prob=prob,
    )
