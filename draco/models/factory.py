"""Model factory for DRACO baselines and RETFound-style ViT."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def build_model(
    name: str,
    num_classes: int = 3,
    pretrained: bool = True,
    checkpoint: str | Path | None = None,
) -> nn.Module:
    name = name.lower().replace("-", "_")

    if name in {"efficientnet_b0", "efficientnetb0", "efficientnet"}:
        return _timm_model("efficientnet_b0", num_classes, pretrained)
    if name in {"resnet50", "resnet_50"}:
        return _timm_model("resnet50", num_classes, pretrained)
    if name in {"retfound", "retfound_vit_large", "vit_large_patch16"}:
        return _build_retfound(num_classes, checkpoint)
    if name.startswith("timm:"):
        return _timm_model(name.split(":", 1)[1], num_classes, pretrained)

    raise ValueError(f"Unknown model: {name}")


def _timm_model(arch: str, num_classes: int, pretrained: bool) -> nn.Module:
    import timm

    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)


def _build_retfound(num_classes: int, checkpoint: str | Path | None) -> nn.Module:
    """ViT-Large patch16 as RETFound backbone.

    If an official RETFound OCT checkpoint is provided, load matching weights
    (strict=False so classification head can be new). Otherwise fall back to
    ImageNet-21k / MAE-compatible ViT-L from timm.
    """
    import timm

    model = timm.create_model(
        "vit_large_patch16_224",
        pretrained=checkpoint is None,
        num_classes=num_classes,
    )

    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.is_absolute():
            # Resolve relative to package/repo root (draco/models -> repo)
            path = Path(__file__).resolve().parents[2] / path
        if not path.exists():
            raise FileNotFoundError(f"RETFound checkpoint not found: {path}")
        # RETFound MAE checkpoints pickle argparse.Namespace in 'args'
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict):
            if "model" in raw:
                state = raw["model"]
            elif "state_dict" in raw:
                state = raw["state_dict"]
            else:
                state = raw
        else:
            state = raw
        # Keep encoder only — drop MAE decoder / mask token
        cleaned = {}
        for k, v in state.items():
            nk = k.replace("module.", "").replace("backbone.", "")
            if nk.startswith("decoder") or nk == "mask_token":
                continue
            cleaned[nk] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(
            f"Loaded RETFound weights from {path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    return model
