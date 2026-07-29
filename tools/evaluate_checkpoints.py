#!/usr/bin/env python3
"""Evaluate one or all MADAv2 checkpoints in an experiment directory."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.list_common import ListSegmentationDataset  # noqa: E402
from models.segmentation_factory import build_segmentor  # noqa: E402
from tools.madav2_acquisition import _NullLogger, load_checkpoint  # noqa: E402


def update_histogram(histogram, prediction, target, num_classes):
    valid = (target >= 0) & (target < num_classes)
    bins = num_classes * target[valid] + prediction[valid]
    histogram += np.bincount(
        bins, minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)


def evaluate(cfg, checkpoint, device, max_samples=None):
    entry = dict(cfg["data"]["target_valid"], shuffle=False)
    dataset = ListSegmentationDataset(
        entry, None, _NullLogger(), augmentations=None
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
    )
    model = build_segmentor(cfg, freeze_bn=True)
    load_checkpoint(model, checkpoint)
    model.to(device).eval()
    num_classes = int(cfg["data"]["n_class"])
    histogram = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.inference_mode():
        for batch_index, (images, labels, _) in enumerate(tqdm(
            loader, desc=checkpoint.name, dynamic_ncols=True
        )):
            if max_samples is not None and batch_index >= max_samples:
                break
            images = images.to(device, non_blocking=True)
            logits = model(images)[-1]
            if logits.shape[-2:] != labels.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            prediction = logits.argmax(dim=1).cpu().numpy()
            target = labels.numpy()
            update_histogram(histogram, prediction, target, num_classes)
    denominator = (
        histogram.sum(axis=1)
        + histogram.sum(axis=0)
        - np.diag(histogram)
    )
    class_iou = np.divide(
        np.diag(histogram),
        denominator,
        out=np.full(num_classes, np.nan, dtype=np.float64),
        where=denominator > 0,
    )
    return {
        "mIoU": float(np.nanmean(class_iou) * 100.0),
        "class_iou": [None if np.isnan(v) else float(v * 100.0) for v in class_iou],
    }


def checkpoint_order(path):
    match = re.search(r"model_iter_(\d+)", path.name)
    if match:
        return (0, int(match.group(1)), path.name)
    return (1, 0, path.name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoints", nargs="*")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Evaluate only this many validation images (for smoke tests).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    run_dir = Path(args.run_dir).expanduser().resolve()
    if args.checkpoints:
        checkpoints = [
            (run_dir / name if not Path(name).is_absolute() else Path(name))
            for name in args.checkpoints
        ]
    else:
        checkpoints = sorted(run_dir.glob("model_*.pth"), key=checkpoint_order)
    if not checkpoints:
        raise FileNotFoundError(f"No model checkpoints found under {run_dir}")
    results = {}
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        results[checkpoint.name] = evaluate(
            config,
            checkpoint,
            torch.device(args.device),
            max_samples=args.max_samples,
        )
        print(f"{checkpoint.name}: {results[checkpoint.name]['mIoU']:.4f} mIoU")
    (run_dir / "evaluation.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
