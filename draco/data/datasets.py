"""PyTorch Dataset from DRACO catalog CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from draco.data.transforms import build_transforms


class CatalogDataset(Dataset):
    """Load OCT images listed in a catalog CSV.

    Expected columns: path, label, label_name, split, dataset
    """

    def __init__(
        self,
        catalog_csv: str | Path,
        transform=None,
        root: str | Path | None = None,
        binary: bool = False,
    ):
        self.df = pd.read_csv(catalog_csv)
        self.root = Path(root) if root else None
        self.transform = transform
        self.binary = binary

        if "path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError(f"Catalog missing path/label columns: {list(self.df.columns)}")

        # Resolve absolute paths
        paths = []
        for p in self.df["path"].astype(str):
            path = Path(p)
            if not path.is_absolute() and self.root is not None:
                path = self.root / path
            paths.append(path)
        self.paths = paths
        labels = self.df["label"].astype(int).tolist()
        if binary:
            # Map 3-class DME -> binary: 0 stay 0, 1+2 -> 1
            labels = [0 if y == 0 else 1 for y in labels]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

    @property
    def num_classes(self) -> int:
        return len(set(self.labels))


def make_loader(
    catalog_csv: str | Path,
    split: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    binary: bool = False,
    class_weights: list[float] | None = None,
    shuffle: bool | None = None,
) -> DataLoader:
    transform = build_transforms(split, image_size=image_size)
    ds = CatalogDataset(catalog_csv, transform=transform, binary=binary)

    use_shuffle = shuffle if shuffle is not None else (split == "train")
    sampler = None
    if split == "train" and class_weights is not None and not binary:
        # WeightedRandomSampler from inverse class frequency of samples
        sample_w = [class_weights[y] for y in ds.labels]
        sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
        use_shuffle = False

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=use_shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
    )
