#!/usr/bin/env python3
"""Run one real-data optimization step and verify checkpoint reloading."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.list_common import ListSegmentationDataset  # noqa: E402
from models.segmentation_factory import (  # noqa: E402
    build_optimizer,
    build_segmentor,
)
from tools.madav2_acquisition import _NullLogger, load_checkpoint  # noqa: E402


def build_loader(entry, workers, batch_size, resize):
    smoke_entry = dict(entry)
    smoke_entry["resize"] = resize
    smoke_entry["batch_size"] = batch_size
    dataset = ListSegmentationDataset(
        smoke_entry,
        None,
        _NullLogger(),
        augmentations=None,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    device = torch.device(args.device)
    workers = min(1, int(cfg["data"].get("num_workers", 0)))
    is_vfm = cfg["model"]["arch"] == "dinov3_base_rein_hrda"
    train_batch_size = 1 if is_vfm else 2
    train_resize = [1024, 2048] if is_vfm else [512, 1024]

    train_loader = build_loader(
        cfg["data"]["source"],
        workers,
        batch_size=train_batch_size,
        resize=train_resize,
    )
    model = build_segmentor(cfg, freeze_bn=False).to(device).train()
    optimizer = build_optimizer(model, cfg)
    images, labels, _ = next(iter(train_loader))
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    logits = model(images)[-1]
    if logits.shape[-2:] != labels.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    loss = F.cross_entropy(logits, labels, ignore_index=255)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite training loss: {loss.item()}")
    loss.backward()
    optimizer.step()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    del optimizer, model, logits
    torch.cuda.empty_cache()

    validation_loader = build_loader(
        cfg["data"]["target_valid"],
        workers,
        batch_size=1,
        resize=[512, 1024],
    )
    reloaded = build_segmentor(cfg, freeze_bn=True)
    load_checkpoint(reloaded, checkpoint)
    reloaded.to(device).eval()
    val_images, _, _ = next(iter(validation_loader))
    with torch.inference_mode():
        val_logits = reloaded(val_images.to(device))[-1]
    if not torch.isfinite(val_logits).all():
        raise RuntimeError("Validation logits contain NaN or Inf")

    report = {
        "model": cfg["model"]["arch"],
        "train_loss": float(loss.item()),
        "train_input_shape": list(images.shape),
        "validation_output_shape": list(val_logits.shape),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "checkpoint": str(checkpoint),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
