"""Model and optimizer factory shared by the MADAv2 training stages."""

import torch

from models.deeplab import DeepLab
from models.madav2_vfm import (
    MADAv2VFMSegmentor,
    WarmupPolyLrScheduler,
    build_vfm_optimizer,
)


def is_vfm(cfg):
    return cfg["model"]["arch"] == "dinov3_base_rein_hrda"


def build_segmentor(cfg, freeze_bn=False):
    if is_vfm(cfg):
        return MADAv2VFMSegmentor(cfg, freeze_bn=freeze_bn)
    return DeepLab(
        num_classes=int(cfg["data"]["n_class"]),
        backbone=cfg["model"]["basenet"]["version"],
        output_stride=int(cfg["model"].get("output_stride", 16)),
        bn=cfg["model"]["bn"],
        freeze_bn=freeze_bn,
    )


def build_optimizer(model, cfg):
    if is_vfm(cfg):
        return build_vfm_optimizer(model, cfg)
    optim_cfg = {
        key: value
        for key, value in cfg["training"]["optimizer"].items()
        if key not in {"name", "head_lr_multiplier"}
    }
    return torch.optim.SGD(
        model.optim_parameters(optim_cfg["lr"]), **optim_cfg
    )


def build_lr_scheduler(model, optimizer, cfg):
    del model
    if is_vfm(cfg):
        return WarmupPolyLrScheduler(
            optimizer,
            max_iter=int(cfg["training"]["train_iters"]),
            warmup_iter=int(cfg["training"].get("warmup_iters", 1500)),
        )
    from schedulers import get_scheduler

    return get_scheduler(optimizer, cfg["training"]["lr_schedule"])
