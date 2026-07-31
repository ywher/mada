#!/usr/bin/env python3
"""Feature extraction, clustering, and image-level acquisition for MADAv2."""

from __future__ import annotations

import argparse
import json
import sys
import time
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


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_checkpoint(model, path):
    path = Path(path).expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model_key = model.__class__.__name__
    if model_key in checkpoint:
        checkpoint = checkpoint[model_key]["model_state"]
    elif "model_state" in checkpoint:
        checkpoint = checkpoint["model_state"]
    model.load_state_dict(checkpoint, strict=False)


def build_dataset(cfg, role):
    entry = cfg["data"][role]
    # Acquisition is deterministic and uses the complete resized image.
    entry = dict(entry, shuffle=False, return_path=True)
    return ListSegmentationDataset(
        entry,
        writer=None,
        logger=_NullLogger(),
        augmentations=None,
    )


class _NullLogger:
    def info(self, *args, **kwargs):
        del args, kwargs


def _class_vectors(features, logits, labels, num_classes, use_gt):
    if logits.shape[-2:] != features.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    prediction = logits.argmax(dim=1)
    if use_gt:
        labels = F.interpolate(
            labels.unsqueeze(1).float(),
            size=features.shape[-2:],
            mode="nearest",
        ).squeeze(1).long()
    vectors = features.new_zeros(
        features.shape[0], num_classes, features.shape[1]
    )
    for batch_index in range(features.shape[0]):
        for class_index in range(num_classes):
            mask = prediction[batch_index] == class_index
            if use_gt:
                mask = mask & (labels[batch_index] == class_index)
            if mask.sum() < 10:
                continue
            vectors[batch_index, class_index] = (
                features[batch_index, :, mask].mean(dim=1)
            )
    return vectors


def extract_features(cfg, role, checkpoint, output_path, device):
    dataset = build_dataset(cfg, role)
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
    vectors = np.zeros(
        (
            len(dataset),
            int(cfg["data"]["n_class"]),
            int(cfg["selection"]["feature_dim"]),
        ),
        dtype=np.float32,
    )
    use_gt = role == "source"
    with torch.inference_mode():
        for index, (images, labels, _) in enumerate(
            tqdm(loader, desc=f"extract-{role}", dynamic_ncols=True)
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            _, _, features, logits = model(images)
            vector = _class_vectors(
                features,
                logits,
                labels,
                int(cfg["data"]["n_class"]),
                use_gt=use_gt,
            )
            vectors[index] = vector[0].cpu().numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, vectors)
    return len(dataset)


def cluster_features(feature_path, output_path, num_centroids, seed):
    features = np.load(feature_path, mmap_mode="r")
    flattened = np.asarray(features.reshape(len(features), -1), dtype=np.float32)
    try:
        import faiss

        kmeans = faiss.Kmeans(
            flattened.shape[1],
            num_centroids,
            niter=50,
            nredo=3,
            seed=seed,
            verbose=True,
            gpu=torch.cuda.is_available(),
        )
        kmeans.train(flattened)
        centroids = kmeans.centroids
        backend = "faiss"
    except ImportError:
        from sklearn.cluster import MiniBatchKMeans

        kmeans = MiniBatchKMeans(
            n_clusters=num_centroids,
            random_state=seed,
            batch_size=min(1024, len(flattened)),
            n_init=3,
            max_iter=100,
        )
        kmeans.fit(flattened)
        centroids = kmeans.cluster_centers_
        backend = "sklearn"
    centroids = centroids.reshape(
        num_centroids, features.shape[1], features.shape[2]
    ).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, centroids)
    return backend


def _nearest_distance(vectors, centroids, batch_size=256):
    vectors = vectors.reshape(len(vectors), -1)
    centroids = centroids.reshape(len(centroids), -1)
    centroid_norm = np.sum(centroids * centroids, axis=1)[None, :]
    output = np.empty(len(vectors), dtype=np.float32)
    for start in range(0, len(vectors), batch_size):
        batch = np.asarray(vectors[start : start + batch_size], dtype=np.float32)
        distances = (
            np.sum(batch * batch, axis=1)[:, None]
            + centroid_norm
            - 2.0 * batch @ centroids.T
        )
        output[start : start + len(batch)] = distances.min(axis=1)
    return output


def select_images(cfg, source_centroids, target_centroids, target_features):
    output_dir = Path(cfg["selection"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors = np.load(target_features, mmap_mode="r")
    source_distance = _nearest_distance(vectors, np.load(source_centroids))
    target_distance = _nearest_distance(vectors, np.load(target_centroids))
    score = source_distance + target_distance
    budget = int(cfg["selection"]["budget"])
    if budget <= 0 or budget > len(score):
        raise ValueError(f"Invalid image budget {budget} for pool {len(score)}")
    selected = np.argpartition(score, -budget)[-budget:]
    selected = selected[np.argsort(-score[selected], kind="stable")]
    selected_sorted = np.sort(selected)

    target_list = Path(cfg["data"]["target"]["list_path"])
    lines = [
        line
        for line in target_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) != len(score):
        raise RuntimeError(
            f"Target list has {len(lines)} samples but features have {len(score)}"
        )
    (output_dir / "selected_indices.txt").write_text(
        "".join(f"{index}\n" for index in selected_sorted),
        encoding="utf-8",
    )
    (output_dir / "selected_images.txt").write_text(
        "".join(f"{lines[index]}\n" for index in selected_sorted),
        encoding="utf-8",
    )
    np.save(output_dir / "selection_scores.npy", score)
    metadata = {
        "pool_size": len(score),
        "budget": budget,
        "ratio": cfg["selection"]["ratio"],
        "unique_selected": int(len(np.unique(selected_sorted))),
        "score": "nearest_source_distance + nearest_target_distance",
    }
    (output_dir / "selection_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def update_timing(output_dir, step, elapsed, timing_file, **metadata):
    path = output_dir / timing_file
    record = json.loads(path.read_text()) if path.is_file() else {"steps": {}}
    record["steps"][step] = {"seconds": elapsed, **metadata}
    record["total_seconds"] = sum(
        value["seconds"] for value in record["steps"].values()
    )
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def timed(
    output_dir,
    name,
    function,
    *args,
    timing_file="acquisition_timing.json",
    **kwargs,
):
    start = time.perf_counter()
    result = function(*args, **kwargs)
    elapsed = time.perf_counter() - start
    metadata = {"result": result} if isinstance(result, (str, int)) else {}
    update_timing(output_dir, name, elapsed, timing_file, **metadata)
    return result


def timed_if_missing(
    artifact,
    output_dir,
    name,
    function,
    *args,
    timing_file="acquisition_timing.json",
    **kwargs,
):
    artifact = Path(artifact)
    if artifact.is_file() and artifact.stat().st_size > 0:
        print(f"Reusing completed acquisition artifact: {artifact}")
        return None
    return timed(
        output_dir,
        name,
        function,
        *args,
        timing_file=timing_file,
        **kwargs,
    )


def run_initial_acquisition(cfg, checkpoint, device):
    output_dir = Path(cfg["selection"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    source_features = output_dir / "source_features.npy"
    target_features = output_dir / "target_warmup_features.npy"
    source_centroids = output_dir / "source_centroids.npy"
    target_centroids = output_dir / "target_warmup_centroids.npy"
    num_centroids = int(cfg["selection"]["num_centroids"])
    seed = int(cfg.get("seed", 1337))
    timed_if_missing(
        source_features,
        output_dir,
        "source_feature_extraction",
        extract_features,
        cfg,
        "source",
        checkpoint,
        source_features,
        device,
    )
    timed_if_missing(
        source_centroids,
        output_dir,
        "source_clustering",
        cluster_features,
        source_features,
        source_centroids,
        num_centroids,
        seed,
    )
    timed_if_missing(
        target_features,
        output_dir,
        "target_feature_extraction",
        extract_features,
        cfg,
        "target",
        checkpoint,
        target_features,
        device,
    )
    timed_if_missing(
        target_centroids,
        output_dir,
        "target_clustering",
        cluster_features,
        target_features,
        target_centroids,
        num_centroids,
        seed,
    )
    timed_if_missing(
        output_dir / "selected_images.txt",
        output_dir,
        "image_selection",
        select_images,
        cfg,
        source_centroids,
        target_centroids,
        target_features,
    )


def run_post_stage1(cfg, checkpoint, device):
    output_dir = Path(cfg["selection"]["output_dir"])
    features = output_dir / "target_stage1_features.npy"
    centroids = output_dir / "target_stage1_centroids.npy"
    timed_if_missing(
        features,
        output_dir,
        "stage1_target_feature_extraction",
        extract_features,
        cfg,
        "target",
        checkpoint,
        features,
        device,
        timing_file="stage2_preparation_timing.json",
    )
    timed_if_missing(
        centroids,
        output_dir,
        "stage1_target_clustering",
        cluster_features,
        features,
        centroids,
        int(cfg["selection"]["num_centroids"]),
        int(cfg.get("seed", 1337)),
        timing_file="stage2_preparation_timing.json",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--phase", choices=["initial", "post-stage1"], default="initial"
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    checkpoint = args.checkpoint
    if not checkpoint:
        key = (
            "source_checkpoint"
            if args.phase == "initial"
            else "stage1_checkpoint"
        )
        checkpoint = config["experiment"][key]
    if args.phase == "initial":
        run_initial_acquisition(config, checkpoint, torch.device(args.device))
    else:
        run_post_stage1(config, checkpoint, torch.device(args.device))
