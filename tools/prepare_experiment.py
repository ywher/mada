#!/usr/bin/env python3
"""Generate portable MADAv2 configs for the five controlled benchmarks."""

from __future__ import annotations

import argparse
import copy
import os
import re
from pathlib import Path

import yaml


DATASETS = {
    "gta2cityscapes": {
        "source_root": "data/gta",
        "source_list": "splits/gta/train.txt",
        "target_root": "data/cityscapes",
        "target_train": "splits/cityscapes/train.txt",
        "target_val": "splits/cityscapes/val.txt",
        "classes": 19,
        "source_class_set": "train_ids",
        "target_class_set": "cityscapes",
        "budget": 47,
        "ratio": "1_64",
    },
    "synthia2cityscapes": {
        "source_root": "data/synthia",
        "source_list": "splits/synthia/train.txt",
        "target_root": "data/cityscapes",
        "target_train": "splits/cityscapes/train.txt",
        "target_val": "splits/cityscapes/val.txt",
        "classes": 16,
        "source_class_set": "train_ids",
        "target_class_set": "syn_city",
        "budget": 47,
        "ratio": "1_64",
    },
    "cityscapes2acdc": {
        "source_root": "data/cityscapes",
        "source_list": "splits/cityscapes/train.txt",
        "target_root": "data/acdc",
        "target_train": "splits/acdc/train.txt",
        "target_val": "splits/acdc/val.txt",
        "classes": 19,
        "source_class_set": "cityscapes",
        "target_class_set": "cityscapes",
        "budget": 25,
        "ratio": "1_64",
    },
    "cityscapes2muses": {
        "source_root": "data/cityscapes",
        "source_list": "splits/cityscapes/train.txt",
        "target_root": "data/muses",
        "target_train": "splits/muses/train.txt",
        "target_val": "splits/muses/val.txt",
        "classes": 19,
        "source_class_set": "cityscapes",
        "target_class_set": "cityscapes",
        "budget": 24,
        "ratio": "1_64",
    },
    "cityscapes2mapillary": {
        "source_root": "data/cityscapes",
        "source_list": "splits/cityscapes/train.txt",
        "target_root": "data/mapillary",
        "target_train": "splits/mapillary/train.txt",
        "target_val": "splits/mapillary/val.txt",
        "classes": 19,
        "source_class_set": "cityscapes",
        "target_class_set": "cityscapes",
        "budget": 141,
        "ratio": "1_128",
    },
}


def _validate_data_pair(root: Path, list_path: Path, split_name: str):
    if not root.is_dir():
        raise FileNotFoundError(f"{split_name} data root does not exist: {root}")
    if not list_path.is_file():
        raise FileNotFoundError(f"{split_name} list does not exist: {list_path}")

    first_pair = None
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            fields = [item for item in re.split(r"[\s,]+", line) if item]
            if len(fields) < 2:
                raise ValueError(
                    f"{split_name} list needs image and label paths: {raw_line}"
                )
            first_pair = fields[:2]
            break
    if first_pair is None:
        raise ValueError(f"{split_name} list is empty: {list_path}")

    for kind, relative_path in zip(("image", "label"), first_pair):
        sample_path = root / relative_path
        if not sample_path.is_file():
            raise FileNotFoundError(
                f"{split_name} first {kind} does not exist: {sample_path}"
            )


def _data_entry(root, list_path, classes, class_set, batch_size, shuffle):
    return {
        "name": "list",
        "rootpath": str(root.absolute()),
        "list_path": str(list_path.resolve()),
        "resize": [1024, 2048],
        "batch_size": batch_size,
        "is_transform": True,
        "shuffle": shuffle,
        "n_class": classes,
        "class_set": class_set,
    }


def build_config(args):
    spec = DATASETS[args.dataset]
    budget = int(args.budget or spec["budget"])
    ratio = args.ratio_name or spec["ratio"]
    repo_root = Path(__file__).resolve().parents[1]
    work_dir = (
        Path(args.output_root).expanduser().resolve()
        / args.dataset
        / ratio
        / args.model
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    source_root = repo_root / spec["source_root"]
    target_root = repo_root / spec["target_root"]
    source_list = repo_root / spec["source_list"]
    target_train = repo_root / spec["target_train"]
    target_val = repo_root / spec["target_val"]
    _validate_data_pair(source_root, source_list, "source train")
    _validate_data_pair(target_root, target_train, "target train")
    _validate_data_pair(target_root, target_val, "target validation")
    pool_size = sum(
        1
        for line in target_train.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if budget > pool_size:
        raise ValueError(f"Budget {budget} exceeds target pool {pool_size}")

    is_vfm = args.model == "dinov3_base_rein_hrda"
    batch_size = args.batch_size or (2 if is_vfm else 4)
    if is_vfm:
        mean = [123.675, 116.28, 103.53]
        std = [58.395, 57.12, 57.375]
        crop = [1024, 1024]
        optimizer = {
            "name": "AdamW",
            "lr": 1.0e-4,
            "weight_decay": 0.05,
            "head_lr_multiplier": 10.0,
        }
    else:
        mean = [0.0, 0.0, 0.0]
        std = [255.0, 255.0, 255.0]
        crop = [1024, 512]
        optimizer = {
            "name": "SGD",
            "lr": 2.5e-4,
            "weight_decay": 5.0e-4,
            "momentum": 0.9,
        }

    if is_vfm:
        vfm_pretrained = repo_root / "pretrained" / "dinov3" / "dinov3_vitb16.pth"
        if not vfm_pretrained.is_file():
            raise FileNotFoundError(
                f"DINOv3 checkpoint does not exist: {vfm_pretrained}. "
                "Run scripts/setup_env.sh first."
            )
    else:
        resnet_pretrained = Path(
            os.environ.get(
                "MADAV2_RESNET101_PRETRAINED",
                repo_root / "pretrained" / "resnet101-5d3b4d8f.pth",
            )
        ).expanduser()
        if not resnet_pretrained.is_file():
            raise FileNotFoundError(
                f"ResNet-101 checkpoint does not exist: {resnet_pretrained}. "
                "Run scripts/setup_env.sh first or set "
                "MADAV2_RESNET101_PRETRAINED."
            )

    source = _data_entry(
        source_root,
        source_list,
        spec["classes"],
        spec["source_class_set"],
        batch_size,
        True,
    )
    target = _data_entry(
        target_root,
        target_train,
        spec["classes"],
        spec["target_class_set"],
        2,
        True,
    )
    target["pool_size"] = pool_size
    target_valid = _data_entry(
        target_root,
        target_val,
        spec["classes"],
        spec["target_class_set"],
        1,
        False,
    )
    for entry in (source, target, target_valid):
        entry["mean"] = mean
        entry["std"] = std

    selected_list = work_dir / "selection" / "selected_images.txt"
    active = copy.deepcopy(target)
    active.update(
        {
            "name": "list_active",
            "active_list_path": str(selected_list),
            "batch_size": batch_size,
            "shuffle": True,
        }
    )

    total_iters = int(args.max_iters)
    stage1_iters = int(args.stage1_iters or total_iters // 2)
    stage2_iters = total_iters - stage1_iters
    if stage1_iters <= 0 or stage2_iters <= 0:
        raise ValueError("Both MADAv2 adaptation stages need positive iterations")

    source_checkpoint = work_dir / "source" / "model_best.pth"
    stage1_checkpoint = work_dir / "stage1" / "model_best.pth"
    config = {
        "seed": args.seed,
        "trainset": "list",
        "valset": "list",
        "experiment": {
            "name": f"{args.dataset}_{args.model}",
            "dataset": args.dataset,
            "ratio": ratio,
            "work_dir": str(work_dir),
            "source_checkpoint": str(source_checkpoint),
            "stage1_checkpoint": str(stage1_checkpoint),
        },
        "selection": {
            "budget": budget,
            "ratio": ratio,
            "num_centroids": 10,
            "feature_dim": 256,
            "output_dir": str(work_dir / "selection"),
        },
        "model": {
            "arch": args.model,
            "pretrained": True,
            "bn": "bn" if is_vfm else "sync_bn",
            "feature_dim": 256,
            "output_stride": 16,
            "default_gpu": 0,
            "init": {"init_type": "kaiming", "init_gain": 0.02},
            "basenet": {"name": "deeplab", "version": "resnet101"},
            "vfm_pretrained": str(
                repo_root / "pretrained" / "dinov3" / "dinov3_vitb16.pth"
            ),
            "vfm_init": args.vfm_init or "",
            "work_dir": str(work_dir),
        },
        "data": {
            "source": source,
            "target": target,
            "active": active,
            "source_valid": None,
            "target_valid": target_valid,
            "num_workers": args.workers,
            "n_class": spec["classes"],
        },
        "training": {
            "epoches": 100000,
            "train_iters": stage1_iters,
            "total_adaptation_iters": total_iters,
            "stage1_iters": stage1_iters,
            "stage2_iters": stage2_iters,
            "val_interval": 10000,
            "print_interval": 100,
            "freeze_bn": False,
            "cls_feature_weight": 0.1,
            "valid_classes": list(range(spec["classes"])),
            "augmentations": {
                "rsize": 2048,
                "rcrop": crop,
                "hflip": 0.5,
            },
            "optimizer": optimizer,
            "loss": {"name": "cross_entropy", "size_average": True},
            "lr_schedule": {
                "name": "poly_lr",
                "gamma": 0.9,
                "max_iter": stage1_iters,
            },
            "warmup_iters": 1500,
            "resume": str(source_checkpoint),
            "Pred_resume": str(source_checkpoint),
            "optimizer_resume": False,
            "gan_resume": False,
            "resume_flag": True,
            "reset_iter_on_resume": True,
        },
        "U2PL": {
            "unsupervised": {
                "TTA": False,
                "drop_percent": 80,
                "apply_aug": "cutmix",
            },
            "contrastive": {
                "negative_high_entropy": True,
                "low_rank": 3,
                "high_rank": 20,
                "current_class_threshold": 0.3,
                "current_class_negative_threshold": 1,
                "unsupervised_entropy_ignore": 80,
                "low_entropy_threshold": 20,
                "num_negatives": 50,
                "num_queries": 256,
                "temperature": 0.5,
            },
        },
        "AEL": {
            "acp": {
                "rand_resize": [0.1, 2.0],
                "prob": 0.5,
                "momentum": 0.999,
                "number": 3,
            },
            "acm": {
                "number": 3,
                "area_thresh": 0.005,
                "area_thresh2": 0.01,
                "no_pad": True,
                "no_slim": True,
            },
            "criterion": {
                "contra_weight": 0.1,
                "consist_weight": 1.0,
                "cons": {"sample": True, "gamma": 2},
                "type": "ohem",
                "kwargs": {"thresh": 0.7, "min_kept": 100000},
            },
        },
    }

    source_cfg = copy.deepcopy(config)
    source_cfg["data"].pop("active")
    source_cfg["training"]["train_iters"] = int(args.source_iters)
    source_cfg["training"]["lr_schedule"]["max_iter"] = int(args.source_iters)
    source_cfg["training"]["resume_flag"] = False
    source_cfg["training"]["resume"] = ""
    source_cfg["training"]["Pred_resume"] = ""

    stage1_cfg = copy.deepcopy(config)

    stage2_cfg = copy.deepcopy(config)
    stage2_cfg["data"]["target"]["name"] = "list_AEL"
    stage2_cfg["data"]["active"]["name"] = "list_active_AEL"
    stage2_cfg["training"]["train_iters"] = stage2_iters
    stage2_cfg["training"]["lr_schedule"]["max_iter"] = stage2_iters
    stage2_cfg["training"]["resume"] = str(stage1_checkpoint)
    stage2_cfg["training"]["Pred_resume"] = str(stage1_checkpoint)

    for name, value in {
        "source.yml": source_cfg,
        "stage1.yml": stage1_cfg,
        "stage2.yml": stage2_cfg,
        "acquisition.yml": config,
    }.items():
        with open(work_dir / name, "w", encoding="utf-8") as handle:
            yaml.safe_dump(value, handle, sort_keys=False)
    print(work_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument(
        "--model",
        default="dinov3_base_rein_hrda",
        choices=["deeplab101", "dinov3_base_rein_hrda"],
    )
    parser.add_argument("--output-root", default="runs/controlled")
    parser.add_argument("--max-iters", type=int, default=40000)
    parser.add_argument("--stage1-iters", type=int)
    parser.add_argument("--source-iters", type=int, default=40000)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--ratio-name")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--vfm-init", default="")
    return parser.parse_args()


if __name__ == "__main__":
    build_config(parse_args())
