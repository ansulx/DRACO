"""Model builders."""

from __future__ import annotations

import torch.nn as nn

from glauco.models.factory import build_backbone, build_model
from glauco.models.fusion import (
    CrossAttentionFusionModel,
    LateFusionModel,
    TokenBackbone,
    TriModalCrossAttentionFusionModel,
    TriModalLateFusionModel,
    build_fusion_model,
)

__all__ = [
    "CrossAttentionFusionModel",
    "LateFusionModel",
    "TokenBackbone",
    "TriModalCrossAttentionFusionModel",
    "TriModalLateFusionModel",
    "build_backbone",
    "build_from_config",
    "build_fusion_model",
    "build_model",
    "is_fusion_config",
    "is_trimodal_config",
]


def is_fusion_config(cfg: dict) -> bool:
    fusion_cfg = cfg.get("fusion") or {}
    return bool(fusion_cfg.get("enabled", False))


def is_trimodal_config(cfg: dict) -> bool:
    fusion_cfg = cfg.get("fusion") or {}
    if not fusion_cfg.get("enabled", False):
        return False
    kind = str(fusion_cfg.get("type", "")).lower().replace("-", "_")
    return bool(fusion_cfg.get("trimodal", False)) or kind.startswith("trimodal")


def build_from_config(
    cfg: dict,
    num_classes: int | None = None,
    load_weights: bool = True,
) -> nn.Module:
    """Build the model described by a training config.

    ``load_weights=False`` skips pretrained/RETFound loading, which is what you
    want before restoring a fine-tuned checkpoint.
    """
    if num_classes is None:
        num_classes = 2 if cfg.get("binary", False) else int(cfg.get("num_classes", 2))

    if is_fusion_config(cfg):
        return build_fusion_model(
            cfg["fusion"], num_classes=num_classes, load_weights=load_weights
        )

    model_cfg = cfg.get("model") or {}
    return build_model(
        model_cfg.get("name", "efficientnet_b0"),
        num_classes=num_classes,
        modality=cfg.get("modality", "oct"),
        pretrained=bool(model_cfg.get("pretrained", True)) and load_weights,
        checkpoint=model_cfg.get("checkpoint") if load_weights else None,
    )
