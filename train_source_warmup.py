#!/usr/bin/env python3
"""Train the common source-supervised initialization used by MADAv2."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from augmentations import get_composed_augmentations
from data.list_common import ListSegmentationDataset
from models.segmentation_factory import (
    build_lr_scheduler,
    build_optimizer,
    build_segmentor,
)
from tensorboardX import SummaryWriter
from utils.utils import get_logger


def save_checkpoint(path, model, optimizer, scheduler, iteration):
    state = {
        model.__class__.__name__: {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        },
        "iter": iteration,
        "best_iou": -1.0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def train(cfg, run_dir, device):
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    logger = get_logger(str(run_dir))
    source_cfg = dict(cfg["data"]["source"], shuffle=True)
    dataset = ListSegmentationDataset(
        source_cfg,
        writer,
        logger,
        augmentations=get_composed_augmentations(
            cfg["training"].get("augmentations")
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(source_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
    )
    model = build_segmentor(cfg, freeze_bn=False).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_lr_scheduler(model, optimizer, cfg)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    max_iters = int(cfg["training"]["train_iters"])
    print_interval = int(cfg["training"].get("print_interval", 100))
    save_interval = int(cfg["training"].get("val_interval", 10000))
    iterator = iter(loader)
    model.train()
    start = time.perf_counter()
    progress = tqdm(range(1, max_iters + 1), desc="source-warmup", dynamic_ncols=True)
    for iteration in progress:
        try:
            images, labels, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, labels, _ = next(iterator)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        _, _, _, logits = model(images)
        logits = F.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if iteration % print_interval == 0:
            elapsed = time.perf_counter() - start
            eta = elapsed / iteration * (max_iters - iteration)
            progress.set_postfix(loss=f"{loss.item():.4f}", eta=f"{eta / 3600:.1f}h")
            logger.info(
                "Iter %d/%d loss=%.6f eta_hours=%.2f",
                iteration,
                max_iters,
                loss.item(),
                eta / 3600,
            )
            writer.add_scalar("loss/source", loss.item(), iteration)
        if iteration % save_interval == 0 and iteration < max_iters:
            save_checkpoint(
                run_dir / f"model_iter_{iteration}.pth",
                model,
                optimizer,
                scheduler,
                iteration,
            )
    final_path = run_dir / "model_final.pth"
    save_checkpoint(final_path, model, optimizer, scheduler, max_iters)
    shutil.copy2(final_path, run_dir / "model_best.pth")
    writer.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    output = (
        Path(args.run_dir)
        if args.run_dir
        else Path(config["experiment"]["work_dir"]) / "source"
    )
    train(config, output.resolve(), torch.device(args.device))
