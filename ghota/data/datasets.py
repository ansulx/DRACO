"""Unified GHOTA catalog dataset (path + remapped 3-class label + disease id)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from ghota.labels import map_label
from ghota.models.factory import disease_to_id

# Image loading: reuse GLAUCO helpers
from glauco.data.datasets import load_image
from glauco.data.transforms import build_transforms


class GhotaCatalogDataset(Dataset):
    """CSV columns: path, label (raw or already mapped), disease, modality.

    If ``remap=True`` (default), ``label`` is treated as dataset-native and
    remapped with ``ghota.labels.map_label`` using ``scheme`` column if present.
    """

    def __init__(
        self,
        catalog_csv: str | Path,
        *,
        diseases: list[str] | None = None,
        image_size: int = 224,
        split: str = "train",
        remap: bool = True,
    ):
        self.df = pd.read_csv(catalog_csv)
        task_col = "disease" if "disease" in self.df.columns else "task"
        self.diseases = diseases or sorted(self.df[task_col].astype(str).unique().tolist())
        self.remap = remap
        self.transform_cache = {}

        labels = []
        disease_ids = []
        modalities = []
        paths = []
        for _, row in self.df.iterrows():
            disease = str(row[task_col])
            scheme = row["scheme"] if "scheme" in self.df.columns else None
            if scheme is None or (isinstance(scheme, float) and pd.isna(scheme)):
                scheme = disease
            raw = int(row["label"])
            y = map_label(raw, disease, scheme=str(scheme)) if remap else raw
            labels.append(y)
            disease_ids.append(disease_to_id(disease, self.diseases))
            modalities.append(str(row.get("modality", "oct")))
            paths.append(Path(str(row["path"])))

        self.paths = paths
        self.labels = labels
        self.disease_ids = disease_ids
        self.modalities = modalities
        self.split = split
        self.image_size = image_size

    def _transform(self, modality: str):
        key = (self.split, modality)
        if key not in self.transform_cache:
            self.transform_cache[key] = build_transforms(
                self.split, self.image_size, modality
            )
        return self.transform_cache[key]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        modality = self.modalities[idx]
        img = load_image(self.paths[idx], modality)
        img = self._transform(modality)(img)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        d = torch.tensor(self.disease_ids[idx], dtype=torch.long)
        return img, d, y
