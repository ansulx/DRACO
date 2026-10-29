"""Modality-aware image transforms for GLAUCO glaucoma training.

Each retinal modality has different symmetry and photometric properties, so the
augmentation policy is chosen per modality rather than shared:

- ``oct``    B-scans encode laterality in the horizontal axis, so horizontal
             flips are unsafe. Mild brightness/contrast jitter is fine.
- ``fundus`` Colour photographs are left/right mirror images of each other, so
             horizontal flips are valid and a stronger photometric policy helps
             with the wide acquisition variability across cameras.
- ``rnflt``  RNFL thickness maps rendered as pseudo-RGB heatmaps: the colour
             channel *is* the measurement, so photometric jitter would corrupt
             the signal. En-face maps are also laterality-specific (no flip).
"""

from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODALITIES = ("oct", "fundus", "rnflt")

_ALIASES = {
    "oct": "oct",
    "bscan": "oct",
    "b_scan": "oct",
    "fundus": "fundus",
    "cfp": "fundus",
    "color_fundus": "fundus",
    "photo": "fundus",
    "slo": "fundus",
    "rnflt": "rnflt",
    "rnfl": "rnflt",
    "thickness": "rnflt",
    "heatmap": "rnflt",
    # Visual-field pattern-deviation maps: treat like thickness heatmaps
    # (no H-flip / no photometric jitter that would corrupt the signal).
    "vf": "rnflt",
    "visual_field": "rnflt",
    "visualfield": "rnflt",
}


def normalize_modality(modality: str | None) -> str:
    """Canonicalise a modality string to one of ``MODALITIES``."""
    key = (modality or "oct").lower().replace("-", "_").strip()
    if key not in _ALIASES:
        raise ValueError(f"Unknown modality: {modality}. Use one of {MODALITIES}.")
    return _ALIASES[key]


def build_transforms(
    split: str,
    image_size: int = 224,
    modality: str = "oct",
) -> transforms.Compose:
    """Train / eval transforms for the given modality."""
    modality = normalize_modality(modality)
    if split != "train":
        return _eval_transforms(image_size)
    if modality == "fundus":
        return _fundus_train(image_size)
    if modality == "rnflt":
        return _rnflt_train(image_size)
    return _oct_train(image_size)


def _eval_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _oct_train(image_size: int) -> transforms.Compose:
    """OCT B-scans: no horizontal flip (laterality), mild photometric jitter."""
    return transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.RandomCrop(image_size),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _fundus_train(image_size: int) -> transforms.Compose:
    """Colour fundus: horizontal flip allowed, stronger geometric/photometric."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size, scale=(0.7, 1.0), ratio=(0.9, 1.11)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
        ]
    )


def _rnflt_train(image_size: int) -> transforms.Compose:
    """RNFL thickness heatmaps: geometry only — colour encodes thickness."""
    return transforms.Compose(
        [
            transforms.Resize(image_size + 16),
            transforms.RandomCrop(image_size),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
