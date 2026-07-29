"""Standalone DINOv3-B + ReIN + HRDA model for MADAv2."""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

from .vfm import HRDAEncoderDecoder, HRDAHead, ReinsDINOv3


def _resolve_path(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return path.resolve()


def _backbone_config():
    return {
        "dinov3_config": {
            "img_size": 512,
            "patch_size": 16,
            "pos_embed_rope_rescale_coords": 2.0,
            "pos_embed_rope_dtype": "fp32",
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "ffn_ratio": 4.0,
            "qkv_bias": True,
            "layerscale_init": 1e-5,
            "ffn_layer": "mlp",
            "ffn_bias": True,
            "proj_bias": True,
            "n_storage_tokens": 4,
            "mask_k_bias": True,
            "out_indices": [2, 5, 8, 11],
        },
        "reins_config": {
            "lora_dim": 16,
            "num_layers": 12,
            "non_adapter_layers": 0,
            "embed_dims": 768,
            "patch_size": 16,
            "token_length": 100,
            "link_token_to_query": False,
        },
    }


def _decoder_config(num_classes):
    return {
        "in_channels": [768, 768, 768, 768],
        "in_index": [0, 1, 2, 3],
        "channels": 256,
        "dropout_ratio": 0.1,
        "num_classes": num_classes,
        "norm_cfg": {"type": "BN", "requires_grad": True},
        "align_corners": False,
        "loss_decode": {
            "type": "CrossEntropyLoss",
            "use_sigmoid": False,
            "loss_weight": 1.0,
        },
        "single_scale_head": "DAFormerHead",
        "interpolate": False,
        "decoder_params": {
            "embed_dims": 256,
            "embed_cfg": {"type": "mlp", "act_cfg": None, "norm_cfg": None},
            "embed_neck_cfg": {
                "type": "mlp",
                "act_cfg": None,
                "norm_cfg": None,
            },
            "fusion_cfg": {
                "type": "aspp",
                "sep": True,
                "dilations": [1, 6, 12, 18],
                "pool": False,
                "act_cfg": {"type": "ReLU"},
                "norm_cfg": {"type": "BN", "requires_grad": True},
            },
        },
        "lr_loss_weight": 0,
        "hr_loss_weight": 0.1,
        "scales": [0.5, 1.0],
        "attention_embed_dim": 256,
        "attention_classwise": True,
        "enable_hr_crop": True,
        "hr_slide_inference": True,
        "hr_slide_overlapping": True,
        "hr_slide_batch_size": 4,
        "crop_coord_divisible": 8,
        "blur_hr_crop": False,
        "feature_scale": 0.5,
        "fixed_attention": None,
        "debug_output_attention": False,
    }


def _extract_logits(output):
    if isinstance(output, dict):
        output = output["seg_logits"]
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output


def _objective_feature(features):
    """Return a deterministic 256-D context feature map for MADAv2 anchors."""
    if isinstance(features, dict):
        features = features["features"]
    if isinstance(features, tuple):
        # ReinsDINOv3 returns (pyramid, original multi-layer features, cls).
        features = features[1]
    if isinstance(features, (tuple, list)):
        features = features[-1]
    if not torch.is_tensor(features):
        raise TypeError(f"Unsupported VFM feature type: {type(features)!r}")
    if features.shape[1] == 768:
        batch, _, height, width = features.shape
        features = features.reshape(batch, 256, 3, height, width).mean(2)
    if features.shape[1] != 256:
        raise ValueError(
            f"MADAv2 requires a 256-D objective feature, got {features.shape}"
        )
    return features


class MADAv2VFMSegmentor(nn.Module):
    """Expose the VFM through MADAv2's four-output DeepLab interface."""

    skip_generic_init = True
    feature_dim = 256

    def __init__(self, cfg, freeze_bn=False):
        super().__init__()
        del freeze_bn
        model_cfg = cfg["model"]
        num_classes = int(cfg["data"]["n_class"])
        pretrained = _resolve_path(model_cfg["vfm_pretrained"])
        if not pretrained.is_file():
            raise FileNotFoundError(f"DINOv3 checkpoint does not exist: {pretrained}")
        backbone = ReinsDINOv3(
            backbone_config=_backbone_config(),
            pretrained={"dinov3": str(pretrained)},
        )
        decode_head = HRDAHead(_decoder_config(num_classes))
        self.segmentor = HRDAEncoderDecoder(
            backbone=backbone,
            decode_head=decode_head,
            auxiliary_head=None,
            token_mask_ratio=None,
            train_cfg={"work_dir": model_cfg.get("work_dir", "runs")},
            test_cfg={
                "mode": "slide",
                "stride": model_cfg.get("slide_stride", [512, 512]),
                "crop_size": model_cfg.get("slide_crop_size", [1024, 1024]),
                "batched_slide": False,
            },
        )
        self.best_iou = 0.0
        init_path = str(model_cfg.get("vfm_init", "") or "").strip()
        if init_path:
            self.load_external_checkpoint(init_path)

    @property
    def model(self):
        return self.segmentor

    def _forward_train(self, images):
        try:
            multires_features, _ = self.model._forward_train_features(images)
            logits = _extract_logits(
                self.model.decode_head.forward_train(multires_features)
            )
            feature = _objective_feature(multires_features[0])
            return feature, logits
        finally:
            self.model.decode_head.reset_crop()

    def _forward_inference(self, images):
        self.model.decode_head.reset_crop()
        output = self.model.encode_decode(
            images, return_feat=True, upscale_pred=True
        )
        return _objective_feature(output["features"]), output["seg_logits"]

    def forward(self, images):
        feature, logits = (
            self._forward_train(images)
            if self.training
            else self._forward_inference(images)
        )
        return feature, feature, feature, logits

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def load_external_checkpoint(self, checkpoint_path):
        checkpoint_path = _resolve_path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if "model_state" in checkpoint:
            checkpoint = checkpoint["model_state"]
        self.load_state_dict(checkpoint, strict=False)
        logging.getLogger(__name__).info(
            "Loaded VFM initialization from %s", checkpoint_path
        )

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        del destination
        state = OrderedDict()
        for name, value in self.model.backbone.adapter.state_dict(
            keep_vars=keep_vars
        ).items():
            state[f"{prefix}adapter.{name}"] = value
        for name, value in self.model.decode_head.state_dict(
            keep_vars=keep_vars
        ).items():
            state[f"{prefix}decoder.{name}"] = value
        return state

    def load_state_dict(self, state_dict, strict=True):
        del strict
        if "adapter" in state_dict or "decoder" in state_dict:
            adapter = state_dict.get("adapter", {})
            decoder = state_dict.get("decoder", {})
        else:
            adapter = {
                key.split("adapter.", 1)[1]: value
                for key, value in state_dict.items()
                if "adapter." in key
            }
            decoder = {
                key.split("decoder.", 1)[1]: value
                for key, value in state_dict.items()
                if "decoder." in key
            }
        adapter_result = self.model.backbone.adapter.load_state_dict(
            adapter, strict=False
        )
        decoder_result = self.model.decode_head.load_state_dict(
            decoder, strict=False
        )
        return adapter_result, decoder_result


class WarmupPolyLrScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        max_iter,
        warmup_iter=1500,
        warmup_ratio=1e-6,
        power=1.0,
        last_epoch=-1,
    ):
        self.max_iter = max_iter
        self.warmup_iter = min(warmup_iter, max_iter)
        self.warmup_ratio = warmup_ratio
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iter:
            alpha = self.last_epoch / max(1, self.warmup_iter)
            ratio = self.warmup_ratio + (1.0 - self.warmup_ratio) * alpha
        else:
            progress = (self.last_epoch - self.warmup_iter) / max(
                1, self.max_iter - self.warmup_iter
            )
            ratio = max(0.0, 1.0 - progress)
        return [base_lr * ratio for base_lr in self.base_lrs]


def build_vfm_optimizer(model, cfg):
    optim_cfg = cfg["training"]["optimizer"]
    base_lr = float(optim_cfg.get("lr", 1e-4))
    base_wd = float(optim_cfg.get("weight_decay", 0.05))
    head_multiplier = float(optim_cfg.get("head_lr_multiplier", 10.0))
    norm_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.GroupNorm,
        nn.LayerNorm,
    )
    groups = []
    seen = set()
    for module_name, module in model.named_modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            full_name = f"{module_name}.{parameter_name}"
            lr = base_lr * (
                head_multiplier if "decode_head" in full_name else 1.0
            )
            no_decay = (
                isinstance(module, norm_types)
                or parameter.ndim == 1
                or "learnable_tokens" in full_name
            )
            groups.append(
                {
                    "params": [parameter],
                    "lr": lr,
                    "weight_decay": 0.0 if no_decay else base_wd,
                }
            )
    return torch.optim.AdamW(
        groups,
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=base_wd,
        eps=1e-8,
    )
