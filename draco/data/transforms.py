"""Shared image transforms for DRACO OCT training."""

from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(split: str, image_size: int = 224) -> transforms.Compose:
    """Train / eval transforms. Horizontal flip off by default (OCT laterality)."""
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize(image_size + 32),
                transforms.RandomCrop(image_size),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
