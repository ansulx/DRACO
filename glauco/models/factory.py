"""Model factory for GLAUCO baselines and RETFound-style ViT backbones."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]

RETFOUND_ARCH = "vit_large_patch16_224"

# Official RETFound weights, per pretraining modality.
DEFAULT_RETFOUND_CHECKPOINTS = {
    "oct": "checkpoints/weights/RETFound_oct.pth",
    "fundus": "checkpoints/weights/RETFound_cfp.pth",
}


def build_model(
    name: str,
    num_classes: int = 2,
    modality: str = "oct",
    pretrained: bool = True,
    checkpoint: str | Path | None = None,
) -> nn.Module:
    """Build a classifier. ``num_classes=0`` returns a pooled feature extractor."""
    key = _canonical(name)

    if key in {"efficientnet_b0", "efficientnetb0", "efficientnet"}:
        return _timm_model("efficientnet_b0", num_classes, pretrained)
    if key in {"resnet50", "resnet_50"}:
        return _timm_model("resnet50", num_classes, pretrained)
    if key in {"retfound", "retfound_vit_large", "vit_large_patch16"}:
        # Plain "retfound" follows the modality: OCT weights for OCT inputs,
        # colour-fundus weights for fundus inputs.
        return _build_retfound(num_classes, checkpoint, modality=modality)
    if key in {"retfound_oct"}:
        return _build_retfound(num_classes, checkpoint, modality="oct")
    if key in {"retfound_cfp", "retfound_fundus", "retfound_color"}:
        return _build_retfound(num_classes, checkpoint, modality="fundus")
    if key.startswith("timm:"):
        return _timm_model(name.split(":", 1)[1], num_classes, pretrained)

    raise ValueError(f"Unknown model: {name}")


def build_backbone(
    name: str,
    modality: str = "oct",
    pretrained: bool = True,
    checkpoint: str | Path | None = None,
) -> tuple[nn.Module, int]:
    """Build a headless backbone and report its feature dimension."""
    model = build_model(
        name,
        num_classes=0,
        modality=modality,
        pretrained=pretrained,
        checkpoint=checkpoint,
    )
    feat_dim = getattr(model, "num_features", None)
    if feat_dim is None:
        raise ValueError(f"Backbone {name} does not expose num_features")
    return model, int(feat_dim)


def _canonical(name: str) -> str:
    name = str(name).strip()
    if name.lower().startswith("timm:"):
        return "timm:" + name.split(":", 1)[1]
    return name.lower().replace("-", "_")


def _timm_model(arch: str, num_classes: int, pretrained: bool) -> nn.Module:
    import timm

    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)


def _resolve_checkpoint(checkpoint: str | Path | None, modality: str) -> Path | None:
    """Explicit path wins; otherwise fall back to the default for the modality."""
    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"RETFound checkpoint not found: {path}")
        return path

    default = DEFAULT_RETFOUND_CHECKPOINTS.get(modality)
    if default is None:
        return None
    path = REPO_ROOT / default
    return path if path.exists() else None


def _build_retfound(
    num_classes: int,
    checkpoint: str | Path | None,
    modality: str = "oct",
) -> nn.Module:
    """ViT-Large patch16 as a RETFound backbone.

    Loads the official RETFound MAE weights for the requested modality when
    available (strict=False so the classification head stays fresh); otherwise
    falls back to the ImageNet-21k / MAE-compatible ViT-L from timm.
    """
    import timm

    from glauco.data.transforms import normalize_modality

    modality = normalize_modality(modality)
    if modality == "rnflt":
        # RNFLT heatmaps are en-face maps — the OCT-pretrained encoder transfers
        # better than the colour-fundus one.
        modality = "oct"

    path = _resolve_checkpoint(checkpoint, modality)
    model = timm.create_model(
        RETFOUND_ARCH,
        pretrained=path is None,
        num_classes=num_classes,
    )
    if path is None:
        return model

    # RETFound MAE checkpoints pickle an argparse.Namespace under 'args'.
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        state = raw.get("model") or raw.get("state_dict") or raw
    else:
        state = raw

    # Keep the encoder only — drop MAE decoder / mask token.
    cleaned = {}
    for k, v in state.items():
        nk = k.replace("module.", "").replace("backbone.", "")
        if nk.startswith("decoder") or nk == "mask_token":
            continue
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(
        f"Loaded RETFound[{modality}] weights from {path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )
    return model
