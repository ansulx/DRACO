"""Build GHOTA models by reusing GLAUCO/DRACO backbone factories."""

from __future__ import annotations

from pathlib import Path

import torch.nn as nn

from ghota.labels import DISEASES, NUM_CLASSES
from ghota.models import GhotaModel

# Reuse existing factories (OCT / fundus / rnflt)
try:
    from glauco.models.factory import build_backbone
except ImportError:  # pragma: no cover
    from draco.models import build_backbone  # type: ignore


def build_ghota_from_config(cfg: dict, load_weights: bool = True) -> nn.Module:
    model_cfg = cfg.get("model") or {}
    name = model_cfg.get("name", "efficientnet_b0")
    modality = cfg.get("modality", "oct")
    diseases = list(cfg.get("diseases") or DISEASES)

    backbone, feat_dim = build_backbone(
        name,
        modality=modality,
        pretrained=bool(model_cfg.get("pretrained", True)) and load_weights,
        checkpoint=model_cfg.get("checkpoint") if load_weights else None,
    )
    # headless: factory may still attach a classifier — strip if present
    if hasattr(backbone, "reset_classifier"):
        backbone.reset_classifier(0)
    elif hasattr(backbone, "fc"):
        backbone.fc = nn.Identity()
    elif hasattr(backbone, "head") and not isinstance(backbone.head, nn.Identity):
        # keep forward_features path; GhotaModel uses it
        pass

    return GhotaModel(
        backbone,
        feat_dim=int(feat_dim),
        num_diseases=len(diseases),
        num_classes=int(cfg.get("num_classes", NUM_CLASSES)),
        disease_dim=int(model_cfg.get("disease_dim", 64)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )


def disease_to_id(name: str, diseases: list[str] | None = None) -> int:
    registry = diseases or list(DISEASES)
    if name not in registry:
        raise KeyError(f"Unknown disease {name!r}; known={registry}")
    return registry.index(name)
