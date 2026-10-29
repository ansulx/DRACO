"""PyTorch Datasets from GLAUCO catalog CSVs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from glauco.data.transforms import build_transforms, normalize_modality

# RNFL thickness (microns) clipping range used to render pseudo-RGB heatmaps.
RNFLT_RANGE = (0.0, 350.0)

_ARRAY_SUFFIXES = {".npy", ".npz"}


def load_image(path: str | Path, modality: str = "oct") -> Image.Image:
    """Open one sample as an RGB PIL image.

    RNFL thickness maps are frequently distributed as raw ``.npy`` / ``.npz``
    arrays; those are colour-mapped into a pseudo-RGB heatmap so the same
    ImageNet-pretrained backbones can consume them.
    """
    path = Path(path)
    if path.suffix.lower() in _ARRAY_SUFFIXES:
        return _array_to_rgb(_load_array(path))
    with Image.open(path) as img:
        return img.convert("RGB")


def _load_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    with np.load(path, allow_pickle=False) as data:
        for key in ("rnflt", "rnfl", "thickness", "arr_0"):
            if key in data:
                return data[key]
        return data[list(data.keys())[0]]


def _array_to_rgb(arr: np.ndarray) -> Image.Image:
    """Normalise a 2-D thickness map and render it through a colormap."""
    arr = np.asarray(arr, dtype=np.float32)
    while arr.ndim > 2:
        arr = arr[..., 0] if arr.shape[-1] <= 4 else arr[0]
    lo, hi = RNFLT_RANGE
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    try:
        import matplotlib

        rgb = matplotlib.colormaps["viridis"](arr)[..., :3]
        rgb = (rgb * 255.0).astype(np.uint8)
    except Exception:  # matplotlib optional — fall back to replicated grayscale
        gray = (arr * 255.0).astype(np.uint8)
        rgb = np.repeat(gray[..., None], 3, axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def _resolve(path: str, root: Path | None) -> Path:
    p = Path(str(path))
    if not p.is_absolute() and root is not None:
        p = root / p
    return p


def _to_binary(labels: list[int]) -> list[int]:
    """Collapse graded glaucoma labels to normal (0) vs any glaucoma (1)."""
    return [0 if y == 0 else 1 for y in labels]


class CatalogDataset(Dataset):
    """Load single-modality images listed in a catalog CSV.

    Expected columns: path, label (plus optional label_name, split, dataset).
    """

    def __init__(
        self,
        catalog_csv: str | Path,
        transform=None,
        root: str | Path | None = None,
        modality: str = "oct",
        binary: bool = False,
        path_col: str = "path",
        label_col: str = "label",
    ):
        self.df = pd.read_csv(catalog_csv)
        self.root = Path(root) if root else None
        self.transform = transform
        self.modality = normalize_modality(modality)
        self.binary = binary

        missing = [c for c in (path_col, label_col) if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"Catalog missing {missing} columns: {list(self.df.columns)}"
            )

        self.paths = [
            _resolve(p, self.root) for p in self.df[path_col].astype(str)
        ]
        labels = self.df[label_col].astype(int).tolist()
        self.labels = _to_binary(labels) if binary else labels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = load_image(self.paths[idx], self.modality)
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

    @property
    def num_classes(self) -> int:
        return len(set(self.labels))


class PairedDataset(Dataset):
    """Load co-registered OCT + fundus pairs for cross-modal fusion.

    Expected columns: oct_path, fundus_path, label.

    Returns ``((oct_tensor, fundus_tensor), label)`` so the default collate
    yields ``[oct_batch, fundus_batch]`` and one training loop can serve both
    single-modality and fusion runs.
    """

    def __init__(
        self,
        catalog_csv: str | Path,
        oct_transform=None,
        fundus_transform=None,
        root: str | Path | None = None,
        binary: bool = False,
        oct_col: str = "oct_path",
        fundus_col: str = "fundus_path",
        label_col: str = "label",
    ):
        self.df = pd.read_csv(catalog_csv)
        self.root = Path(root) if root else None
        self.oct_transform = oct_transform
        self.fundus_transform = fundus_transform
        self.binary = binary

        missing = [
            c for c in (oct_col, fundus_col, label_col) if c not in self.df.columns
        ]
        if missing:
            raise ValueError(
                f"Paired catalog missing {missing} columns: {list(self.df.columns)}"
            )

        self.oct_paths = [_resolve(p, self.root) for p in self.df[oct_col].astype(str)]
        self.fundus_paths = [
            _resolve(p, self.root) for p in self.df[fundus_col].astype(str)
        ]
        labels = self.df[label_col].astype(int).tolist()
        self.labels = _to_binary(labels) if binary else labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        oct_img = load_image(self.oct_paths[idx], "oct")
        fundus_img = load_image(self.fundus_paths[idx], "fundus")
        if self.oct_transform is not None:
            oct_img = self.oct_transform(oct_img)
        if self.fundus_transform is not None:
            fundus_img = self.fundus_transform(fundus_img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return (oct_img, fundus_img), label

    @property
    def num_classes(self) -> int:
        return len(set(self.labels))


def _build_sampler(
    labels: list[int],
    class_weights: list[float] | None,
    balance: bool,
) -> WeightedRandomSampler | None:
    """Weighted sampler from explicit weights or inverse class frequency."""
    if class_weights is None and not balance:
        return None
    if class_weights is None:
        counts = Counter(labels)
        n_classes = max(labels) + 1 if labels else 0
        class_weights = [
            len(labels) / (len(counts) * max(counts.get(c, 0), 1))
            for c in range(n_classes)
        ]
    sample_w = [float(class_weights[y]) for y in labels]
    return WeightedRandomSampler(
        sample_w, num_samples=len(sample_w), replacement=True
    )


def _loader(ds: Dataset, split: str, batch_size: int, num_workers: int, sampler, shuffle):
    use_shuffle = shuffle if shuffle is not None else (split == "train")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False if sampler is not None else use_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
        persistent_workers=num_workers > 0,
    )


def make_loader(
    catalog_csv: str | Path,
    split: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    modality: str = "oct",
    binary: bool = False,
    class_weights: list[float] | None = None,
    balance: bool = False,
    root: str | Path | None = None,
    shuffle: bool | None = None,
) -> DataLoader:
    """DataLoader over a single-modality catalog."""
    transform = build_transforms(split, image_size=image_size, modality=modality)
    ds = CatalogDataset(
        catalog_csv,
        transform=transform,
        root=root,
        modality=modality,
        binary=binary,
    )
    sampler = (
        _build_sampler(ds.labels, class_weights, balance)
        if split == "train"
        else None
    )
    return _loader(ds, split, batch_size, num_workers, sampler, shuffle)


def make_paired_loader(
    catalog_csv: str | Path,
    split: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    binary: bool = False,
    class_weights: list[float] | None = None,
    balance: bool = False,
    root: str | Path | None = None,
    shuffle: bool | None = None,
) -> DataLoader:
    """DataLoader over a paired OCT/fundus catalog for fusion training."""
    ds = PairedDataset(
        catalog_csv,
        oct_transform=build_transforms(split, image_size=image_size, modality="oct"),
        fundus_transform=build_transforms(
            split, image_size=image_size, modality="fundus"
        ),
        root=root,
        binary=binary,
    )
    sampler = (
        _build_sampler(ds.labels, class_weights, balance)
        if split == "train"
        else None
    )
    return _loader(ds, split, batch_size, num_workers, sampler, shuffle)


class TriModalDataset(Dataset):
    """Load co-registered OCT + fundus + visual-field triples (GLEAM).

    Expected columns: oct_path, fundus_path, vf_path, label.
    Returns ``((oct, fundus, vf), label)``.
    """

    def __init__(
        self,
        catalog_csv: str | Path,
        oct_transform=None,
        fundus_transform=None,
        vf_transform=None,
        root: str | Path | None = None,
        binary: bool = False,
        oct_col: str = "oct_path",
        fundus_col: str = "fundus_path",
        vf_col: str = "vf_path",
        label_col: str = "label",
    ):
        self.df = pd.read_csv(catalog_csv)
        self.root = Path(root) if root else None
        self.oct_transform = oct_transform
        self.fundus_transform = fundus_transform
        self.vf_transform = vf_transform
        self.binary = binary

        missing = [
            c
            for c in (oct_col, fundus_col, vf_col, label_col)
            if c not in self.df.columns
        ]
        if missing:
            raise ValueError(
                f"Tri-modal catalog missing {missing}: {list(self.df.columns)}"
            )

        self.oct_paths = [_resolve(p, self.root) for p in self.df[oct_col].astype(str)]
        self.fundus_paths = [
            _resolve(p, self.root) for p in self.df[fundus_col].astype(str)
        ]
        self.vf_paths = [_resolve(p, self.root) for p in self.df[vf_col].astype(str)]
        labels = self.df[label_col].astype(int).tolist()
        self.labels = _to_binary(labels) if binary else labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        oct_img = load_image(self.oct_paths[idx], "oct")
        fundus_img = load_image(self.fundus_paths[idx], "fundus")
        vf_img = load_image(self.vf_paths[idx], "vf")
        if self.oct_transform is not None:
            oct_img = self.oct_transform(oct_img)
        if self.fundus_transform is not None:
            fundus_img = self.fundus_transform(fundus_img)
        if self.vf_transform is not None:
            vf_img = self.vf_transform(vf_img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return (oct_img, fundus_img, vf_img), label

    @property
    def num_classes(self) -> int:
        return len(set(self.labels))


def make_trimodal_loader(
    catalog_csv: str | Path,
    split: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    binary: bool = False,
    class_weights: list[float] | None = None,
    balance: bool = False,
    root: str | Path | None = None,
    shuffle: bool | None = None,
) -> DataLoader:
    """DataLoader over a tri-modal OCT/fundus/VF catalog."""
    ds = TriModalDataset(
        catalog_csv,
        oct_transform=build_transforms(split, image_size=image_size, modality="oct"),
        fundus_transform=build_transforms(
            split, image_size=image_size, modality="fundus"
        ),
        vf_transform=build_transforms(split, image_size=image_size, modality="vf"),
        root=root,
        binary=binary,
    )
    sampler = (
        _build_sampler(ds.labels, class_weights, balance)
        if split == "train"
        else None
    )
    return _loader(ds, split, batch_size, num_workers, sampler, shuffle)
