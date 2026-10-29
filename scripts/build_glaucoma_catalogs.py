#!/usr/bin/env python3
"""Build processed catalog CSVs for GLAUCO glaucoma datasets."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "glaucoma"
OUT = ROOT / "data" / "processed" / "glaucoma" / "catalogs"
STATS = ROOT / "data" / "processed" / "glaucoma" / "stats"


def _class_weights(labels: list[int], num_classes: int) -> list[float]:
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for c in range(num_classes):
        n = counts.get(c, 0)
        weights.append(total / (num_classes * max(n, 1)))
    return weights


def _write_split(catalog: pd.DataFrame, name: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    train = catalog[catalog["split"] == "train"]
    test = catalog[catalog["split"] == "test"]
    train_path = OUT / f"{name}_train.csv"
    test_path = OUT / f"{name}_test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    labels = train["label"].astype(int).tolist()
    num_classes = int(catalog["label"].max()) + 1
    weights = _class_weights(labels, num_classes)
    STATS.mkdir(parents=True, exist_ok=True)
    stats_path = STATS / f"{name}_class_weights.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({"weights": weights, "counts": dict(Counter(labels))}, f, indent=2)

    return {
        "train": len(train),
        "test": len(test),
        "train_csv": str(train_path.relative_to(ROOT)),
        "test_csv": str(test_path.relative_to(ROOT)),
        "class_weights": str(stats_path.relative_to(ROOT)),
    }


def build_harvard_gd(seed: int = 42) -> dict:
    """Harvard-GD: 500 RNFLT maps with binary glaucoma labels.

    HuggingFace layout stores stacked arrays:
      Dataset/rnflt_map.npy       (N, 225, 225)
      Dataset/glaucoma_label.npy  (N,)
    We materialize per-sample .npy slices for the catalog loader.
    """
    import numpy as np
    from sklearn.model_selection import train_test_split

    root = RAW / "harvard_gd"
    if not root.exists():
        raise FileNotFoundError(
            f"Missing {root}. Run: python scripts/download_glaucoma_datasets.py --dataset harvard_gd"
        )

    stacked_rnflt = root / "Dataset" / "rnflt_map.npy"
    stacked_labels = root / "Dataset" / "glaucoma_label.npy"
    if stacked_rnflt.exists() and stacked_labels.exists():
        rnflt = np.load(stacked_rnflt)
        labels = np.load(stacked_labels).astype(int)
        slice_dir = ROOT / "data" / "processed" / "glaucoma" / "harvard_gd_slices"
        slice_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for i in range(len(labels)):
            out_path = slice_dir / f"sample_{i:04d}.npy"
            if not out_path.exists():
                np.save(out_path, rnflt[i].astype(np.float32))
            label = int(labels[i])
            rows.append(
                {
                    "path": str(out_path.resolve()),
                    "label": label,
                    "label_name": "glaucoma" if label else "normal",
                    "split": "unknown",
                    "dataset": "harvard_gd",
                    "modality": "rnflt",
                }
            )
        catalog = pd.DataFrame(rows)
        idx = catalog.index.tolist()
        lbls = catalog["label"].tolist()
        train_idx, test_idx = train_test_split(
            idx, test_size=0.2, random_state=seed, stratify=lbls
        )
        catalog.loc[train_idx, "split"] = "train"
        catalog.loc[test_idx, "split"] = "test"
        print(f"Harvard-GD: {len(rows)} RNFLT slices ({sum(lbls)} glaucoma)")
        return _write_split(catalog, "harvard_gd")

    # Fallback: scan for loose image files (other Harvard releases)
    rows = []
    image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy", ".npz"}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in image_exts:
            continue
        if "rnflt_map.npy" in path.name or "glaucoma_label" in path.name:
            continue
        label = _infer_harvard_label(path, None)
        if label is None:
            continue
        rows.append(
            {
                "path": str(path.resolve()),
                "label": label,
                "label_name": "glaucoma" if label == 1 else "normal",
                "split": "unknown",
                "dataset": "harvard_gd",
                "modality": "rnflt",
            }
        )

    if not rows:
        raise RuntimeError(f"No labeled images found under {root}")

    catalog = pd.DataFrame(rows)
    idx = catalog.index.tolist()
    lbls = catalog["label"].tolist()
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=seed, stratify=lbls
    )
    catalog.loc[train_idx, "split"] = "train"
    catalog.loc[test_idx, "split"] = "test"
    return _write_split(catalog, "harvard_gd")


def build_harvard_gdp(seed: int = 42) -> dict:
    """Harvard-GDP: 1,000 patients — per-patient RNFLT .npz or stacked .npy."""
    import numpy as np
    from sklearn.model_selection import train_test_split

    root = RAW / "harvard_gdp"
    rnflt_dir = root / "Dataset" / "extracted" / "RNFLT"
    if not rnflt_dir.exists():
        rnflt_dir = next((p for p in root.rglob("RNFLT") if p.is_dir()), None)

    rows = []
    if rnflt_dir and rnflt_dir.exists():
        for npz_path in sorted(rnflt_dir.glob("data_*.npz")):
            with np.load(npz_path, allow_pickle=False) as data:
                label = int(data["glaucoma"].item())
            rows.append(
                {
                    "path": str(npz_path.resolve()),
                    "label": label,
                    "label_name": "glaucoma" if label else "normal",
                    "split": "unknown",
                    "dataset": "harvard_gdp",
                    "modality": "rnflt",
                }
            )
    else:
        stacked_rnflt = root / "Dataset" / "rnflt_map.npy"
        stacked_labels = root / "Dataset" / "glaucoma_label.npy"
        if not stacked_rnflt.exists():
            candidates = list(root.rglob("rnflt_map.npy"))
            if candidates:
                stacked_rnflt = candidates[0]
                stacked_labels = stacked_rnflt.parent / "glaucoma_label.npy"
            else:
                raise FileNotFoundError(
                    f"Missing Harvard-GDP under {root}. "
                    "Run: python scripts/download_glaucoma_datasets.py --dataset harvard_gdp"
                )
        rnflt = np.load(stacked_rnflt)
        labels = np.load(stacked_labels).astype(int)
        slice_dir = ROOT / "data" / "processed" / "glaucoma" / "harvard_gdp_slices"
        slice_dir.mkdir(parents=True, exist_ok=True)
        for i in range(len(labels)):
            out_path = slice_dir / f"sample_{i:04d}.npy"
            if not out_path.exists():
                np.save(out_path, rnflt[i].astype(np.float32))
            label = int(labels[i])
            rows.append(
                {
                    "path": str(out_path.resolve()),
                    "label": label,
                    "label_name": "glaucoma" if label else "normal",
                    "split": "unknown",
                    "dataset": "harvard_gdp",
                    "modality": "rnflt",
                }
            )

    if not rows:
        raise RuntimeError(f"No Harvard-GDP samples found under {root}")

    catalog = pd.DataFrame(rows)
    idx = catalog.index.tolist()
    lbls = catalog["label"].tolist()
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=seed, stratify=lbls
    )
    catalog.loc[train_idx, "split"] = "train"
    catalog.loc[test_idx, "split"] = "test"
    print(f"Harvard-GDP: {len(rows)} RNFLT samples ({sum(lbls)} glaucoma)")
    return _write_split(catalog, "harvard_gdp")


def _infer_harvard_label(path: Path, label_df: pd.DataFrame | None) -> int | None:
    """Map image path to binary label from CSV or folder structure."""
    parts = [p.lower() for p in path.parts]
    if "glaucoma" in parts or "pos" in parts or "positive" in parts:
        return 1
    if "normal" in parts or "neg" in parts or "negative" in parts or "control" in parts:
        return 0

    if label_df is not None:
        cols = {c.lower(): c for c in label_df.columns}
        id_col = next(
            (cols[k] for k in ("id", "patient_id", "filename", "file", "image", "name") if k in cols),
            None,
        )
        label_col = next(
            (cols[k] for k in ("glaucoma", "label", "class", "diagnosis") if k in cols),
            None,
        )
        if id_col and label_col:
            stem = path.stem.lower()
            for _, row in label_df.iterrows():
                rid = str(row[id_col]).lower()
                if stem in rid or rid in stem or stem == Path(rid).stem.lower():
                    val = row[label_col]
                    if isinstance(val, str):
                        return 1 if val.lower() in ("1", "glaucoma", "pos", "yes") else 0
                    return int(val)
    # Heuristic: EyeLearn often names glaucoma files with 'g' prefix patterns
    name = path.stem.lower()
    if name.startswith("g") and not name.startswith("gd"):
        return 1
    return 0  # default normal if in dataset root


def build_airogs_light(seed: int = 42) -> dict:
    """AIROGS-light-v2: balanced RG/NRG fundus (Kaggle or local eyepac-light folder)."""
    candidates = [
        RAW / "airogs_light",
        ROOT / "eyepac-light-v2-512-jpg",
    ]
    root = next((p for p in candidates if p.exists()), None)
    if root is None:
        raise FileNotFoundError(
            f"Missing AIROGS-light data. Expected one of:\n"
            f"  {RAW / 'airogs_light'}\n"
            f"  {ROOT / 'eyepac-light-v2-512-jpg'}\n"
            "Download: https://www.kaggle.com/datasets/deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2"
        )

    # Official splits: train / validation / test. Use validation for training early-stop.
    folder_splits = [
        ("train", "train"),
        ("validation", "test"),
        ("test", "holdout"),
    ]
    rows = []
    for folder_name, split_name in folder_splits:
        base = root / folder_name
        if not base.exists():
            continue
        for label_dir, label in [("RG", 1), ("NRG", 0)]:
            label_path = base / label_dir
            if not label_path.exists():
                continue
            for path in sorted(label_path.glob("*.jpg")):
                rows.append(
                    {
                        "path": str(path.resolve()),
                        "label": label,
                        "label_name": "referable_glaucoma" if label else "no_referable_glaucoma",
                        "split": split_name,
                        "dataset": "airogs_light",
                        "modality": "fundus",
                    }
                )

    if not rows:
        raise RuntimeError(f"No images found under {root}")

    catalog = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    train = catalog[catalog["split"] == "train"]
    test = catalog[catalog["split"] == "test"]
    holdout = catalog[catalog["split"] == "holdout"]
    train.to_csv(OUT / "airogs_light_train.csv", index=False)
    test.to_csv(OUT / "airogs_light_test.csv", index=False)
    if len(holdout):
        holdout.to_csv(OUT / "airogs_light_holdout_test.csv", index=False)

    labels = train["label"].astype(int).tolist()
    weights = _class_weights(labels, 2)
    STATS.mkdir(parents=True, exist_ok=True)
    stats_path = STATS / "airogs_light_class_weights.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({"weights": weights, "counts": dict(Counter(labels))}, f, indent=2)

    print(
        f"AIROGS-light from {root.name}: "
        f"train={len(train)}, val(early-stop)={len(test)}, holdout={len(holdout)}"
    )
    return {
        "train": len(train),
        "test": len(test),
        "holdout": len(holdout),
        "train_csv": str((OUT / "airogs_light_train.csv").relative_to(ROOT)),
        "test_csv": str((OUT / "airogs_light_test.csv").relative_to(ROOT)),
        "class_weights": str(stats_path.relative_to(ROOT)),
    }


_GLEAM_LABEL_NAMES = ("normal", "early", "intermediate", "advanced")


def _find_gleam_root() -> Path:
    candidates = [
        ROOT / "GLEAM",
        RAW / "gleam",
        ROOT / "data" / "raw" / "glaucoma" / "gleam",
    ]
    root = next((p for p in candidates if (p / "split" / "trainset").exists()), None)
    if root is None:
        raise FileNotFoundError(
            "Missing GLEAM data. Expected GLEAM/ at repo root or data/raw/glaucoma/gleam/\n"
            "Download: https://www.kaggle.com/datasets/zhangyiyinge/gleam-dataset"
        )
    return root


def build_gleam_paired(seed: int = 42) -> dict:
    """GLEAM: 1,200 paired fundus (SLO) + cpOCT thickness, 4-stage grading.

    Uses the official train/val/test splits under ``split/``. The ``*mis``
    folders are for missing-modality benchmarks and are not included here.
    """
    root = _find_gleam_root()
    split_specs = [
        ("trainset", "trainsetlabel.csv", "train"),
        ("valset", "valsetlabel.csv", "val"),
        ("testset", "testsetlabel.csv", "test"),
    ]
    rows = []
    for folder, label_file, split_name in split_specs:
        label_path = root / "split" / label_file
        if not label_path.exists():
            raise FileNotFoundError(f"Missing {label_path}")
        labels_df = pd.read_csv(label_path)
        id_col, label_col = labels_df.columns[0], labels_df.columns[1]
        label_lookup = {
            int(row[id_col]): int(row[label_col]) for _, row in labels_df.iterrows()
        }
        split_dir = root / "split" / folder
        for sample_dir in sorted(split_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            sample_id = int(sample_dir.name)
            label = label_lookup.get(sample_id)
            if label is None:
                continue
            fundus = sample_dir / "SLO.jpg"
            oct_path = sample_dir / "Thickness.jpg"
            vf_path = sample_dir / "VF.jpg"
            if not fundus.exists() or not oct_path.exists():
                continue
            rows.append(
                {
                    "sample_id": sample_id,
                    "oct_path": str(oct_path.resolve()),
                    "fundus_path": str(fundus.resolve()),
                    "vf_path": str(vf_path.resolve()) if vf_path.exists() else "",
                    "label": label,
                    "label_name": _GLEAM_LABEL_NAMES[label],
                    "split": split_name,
                    "dataset": "gleam",
                }
            )

    if not rows:
        raise RuntimeError(f"No paired GLEAM samples under {root / 'split'}")

    catalog = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    for split_name, out_name in [
        ("train", "gleam_paired_train"),
        ("val", "gleam_paired_val"),
        ("test", "gleam_paired_test"),
    ]:
        part = catalog[catalog["split"] == split_name]
        part.to_csv(OUT / f"{out_name}.csv", index=False)

    train_labels = catalog[catalog["split"] == "train"]["label"].astype(int).tolist()
    weights = _class_weights(train_labels, 4)
    STATS.mkdir(parents=True, exist_ok=True)
    stats_path = STATS / "gleam_paired_class_weights.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({"weights": weights, "counts": dict(Counter(train_labels))}, f, indent=2)

    counts = catalog.groupby("split").size().to_dict()
    print(f"GLEAM paired catalog from {root}: {counts}")
    return {
        "splits": counts,
        "train_csv": str((OUT / "gleam_paired_train.csv").relative_to(ROOT)),
        "val_csv": str((OUT / "gleam_paired_val.csv").relative_to(ROOT)),
        "test_csv": str((OUT / "gleam_paired_test.csv").relative_to(ROOT)),
        "class_weights": str(stats_path.relative_to(ROOT)),
    }


def build_gamma_paired(seed: int = 42) -> dict:
    root = RAW / "gamma"
    if not root.exists():
        raise FileNotFoundError(f"Missing {root}. Register at gamma.grand-challenge.org")

    # GAMMA: Training400 + Validation100 + Test100 with paired fundus + OCT
    rows = []
    for split_dir, split_name in [
        ("Training400", "train"),
        ("Validation100", "test"),
        ("Test100", "test"),
    ]:
        base = root / split_dir
        if not base.exists():
            continue
        for fundus in base.rglob("*-1.jpg"):
            # GAMMA naming: XXXXXX-1.jpg = fundus, XXXXXX-2 = OCT projection
            stem = fundus.stem.replace("-1", "")
            oct_candidates = list(fundus.parent.glob(f"{stem}-2*"))
            if not oct_candidates:
                continue
            label_path = fundus.parent / f"{stem}.txt"
            if label_path.exists():
                label = int(label_path.read_text().strip())
            else:
                continue
            rows.append(
                {
                    "oct_path": str(oct_candidates[0].resolve()),
                    "fundus_path": str(fundus.resolve()),
                    "label": label,
                    "label_name": ["normal", "early", "advanced"][label],
                    "split": split_name,
                    "dataset": "gamma",
                }
            )

    if not rows:
        raise RuntimeError(f"No paired GAMMA samples under {root}")

    catalog = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    train = catalog[catalog["split"] == "train"]
    test = catalog[catalog["split"] == "test"]
    train.to_csv(OUT / "gamma_paired_train.csv", index=False)
    test.to_csv(OUT / "gamma_paired_test.csv", index=False)
    return {"train": len(train), "test": len(test)}


def build_harvard_gf(seed: int = 42) -> dict:
    """Harvard-GF: 3,300 patients with official train/val/test from data_summary.csv."""
    root = RAW / "harvard_gf"
    summary = root / "ReadMe" / "data_summary.csv"
    if not summary.exists():
        raise FileNotFoundError(
            f"Missing {summary}. Run: python scripts/download_glaucoma_datasets.py --dataset harvard_gf"
        )

    extracted = _ensure_gf_extracted(root)
    meta = pd.read_csv(summary)
    split_map = {"training": "train", "validation": "val", "test": "test"}
    rows = []
    missing = 0
    for _, row in meta.iterrows():
        fname = str(row["filename"])
        use = str(row["use"]).strip().lower()
        npz_path = _gf_npz_path(extracted, use, fname)
        if not npz_path.exists():
            missing += 1
            continue
        glaucoma = str(row["glaucoma"]).strip().lower()
        label = 1 if glaucoma in ("yes", "1", "glaucoma") else 0
        split = split_map.get(use, "unknown")
        rows.append(
            {
                "path": str(npz_path.resolve()),
                "label": label,
                "label_name": "glaucoma" if label else "normal",
                "split": split,
                "dataset": "harvard_gf",
                "modality": "rnflt",
                "race": row.get("race", ""),
                "gender": row.get("gender", ""),
            }
        )

    if not rows:
        raise RuntimeError(
            f"No Harvard-GF .npz files under {extracted}. "
            "Check Dataset/extracted/Training|Validation|Test."
        )
    if missing:
        print(f"Warning: {missing} entries in data_summary.csv missing on disk")

    catalog = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    for split_name, out_name in [("train", "harvard_gf_train"), ("val", "harvard_gf_val"), ("test", "harvard_gf_test")]:
        part = catalog[catalog["split"] == split_name]
        part.to_csv(OUT / f"{out_name}.csv", index=False)

    train_labels = catalog[catalog["split"] == "train"]["label"].astype(int).tolist()
    weights = _class_weights(train_labels, 2)
    STATS.mkdir(parents=True, exist_ok=True)
    stats_path = STATS / "harvard_gf_class_weights.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({"weights": weights, "counts": dict(Counter(train_labels))}, f, indent=2)

    counts = catalog.groupby("split").size().to_dict()
    print(f"Harvard-GF RNFLT catalog: {counts}")
    return {
        "splits": counts,
        "train_csv": str((OUT / "harvard_gf_train.csv").relative_to(ROOT)),
        "val_csv": str((OUT / "harvard_gf_val.csv").relative_to(ROOT)),
        "test_csv": str((OUT / "harvard_gf_test.csv").relative_to(ROOT)),
        "class_weights": str(stats_path.relative_to(ROOT)),
    }


def _ensure_gf_extracted(root: Path) -> Path:
    """Extract dataset.zip if needed; return extracted/ base (Training|Validation|Test)."""
    import zipfile

    extract_to = root / "Dataset" / "extracted"
    if (extract_to / "Training").exists() and any((extract_to / "Training").glob("data_*.npz")):
        return extract_to

    zpath = root / "Dataset" / "dataset.zip"
    if not zpath.exists():
        raise FileNotFoundError(f"Missing {zpath}. Run download_glaucoma_datasets.py --dataset harvard_gf")

    if not any(extract_to.rglob("data_*.npz")):
        print(f"Extracting {zpath} -> {extract_to} (may take several minutes)...")
        extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(extract_to)
    return extract_to


_GF_USE_DIRS = {
    "training": "Training",
    "validation": "Validation",
    "test": "Test",
}


def _gf_npz_path(extracted: Path, use: str, fname: str) -> Path:
    sub = _GF_USE_DIRS.get(use.lower())
    if sub:
        path = extracted / sub / fname
        if path.exists():
            return path
    for sub in _GF_USE_DIRS.values():
        path = extracted / sub / fname
        if path.exists():
            return path
    return extracted / fname


def build_acrima(seed: int = 42) -> dict:
    """ACRIMA: 705 OD-cropped fundus images (396 glaucoma / 309 normal).

    Labels from filename: ``_g_`` = glaucoma, else normal.
    """
    from sklearn.model_selection import train_test_split

    root = RAW / "acrima" / "Database" / "Images"
    if not root.exists():
        raise FileNotFoundError(
            f"Missing {root}. Place ACRIMA under data/raw/glaucoma/acrima/"
        )

    rows = []
    for path in sorted(root.glob("*.jpg")):
        label = 1 if "_g_" in path.name else 0
        rows.append(
            {
                "path": str(path.resolve()),
                "label": label,
                "label_name": "glaucoma" if label else "normal",
                "split": "unknown",
                "dataset": "acrima",
                "modality": "fundus",
            }
        )
    if not rows:
        raise RuntimeError(f"No ACRIMA images under {root}")

    catalog = pd.DataFrame(rows)
    idx = catalog.index.tolist()
    lbls = catalog["label"].tolist()
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=seed, stratify=lbls
    )
    catalog.loc[train_idx, "split"] = "train"
    catalog.loc[test_idx, "split"] = "test"
    print(f"ACRIMA: {len(rows)} fundus ({sum(lbls)} glaucoma / {len(rows) - sum(lbls)} normal)")
    return _write_split(catalog, "acrima")


def build_beh(seed: int = 42) -> dict:
    """BEH (Bangladesh Eye Hospital): folder-labeled glaucoma vs normal fundus."""
    from sklearn.model_selection import train_test_split

    candidates = list((RAW / "beh").rglob("glaucoma"))
    glauc_dir = next(
        (p for p in candidates if p.is_dir() and (p.parent / "normal").is_dir()),
        None,
    )
    if glauc_dir is None:
        raise FileNotFoundError(
            f"Missing BEH glaucoma/normal folders under {RAW / 'beh'}"
        )
    normal_dir = glauc_dir.parent / "normal"

    rows = []
    for label, folder in [(1, glauc_dir), (0, normal_dir)]:
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
                continue
            if "__MACOSX" in path.parts:
                continue
            rows.append(
                {
                    "path": str(path.resolve()),
                    "label": label,
                    "label_name": "glaucoma" if label else "normal",
                    "split": "unknown",
                    "dataset": "beh",
                    "modality": "fundus",
                }
            )
    if not rows:
        raise RuntimeError(f"No BEH images under {glauc_dir.parent}")

    catalog = pd.DataFrame(rows)
    idx = catalog.index.tolist()
    lbls = catalog["label"].tolist()
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=seed, stratify=lbls
    )
    catalog.loc[train_idx, "split"] = "train"
    catalog.loc[test_idx, "split"] = "test"
    print(
        f"BEH: {len(rows)} fundus "
        f"({sum(lbls)} glaucoma / {len(rows) - sum(lbls)} normal)"
    )
    return _write_split(catalog, "beh")


def build_rim_one(seed: int = 42) -> dict:
    """RIM-ONE DL: official by-hospital train/test fundus split (domain shift)."""
    root = (
        RAW
        / "rim_one"
        / "RIM-ONE_DL_images"
        / "partitioned_by_hospital"
    )
    if not root.exists():
        raise FileNotFoundError(
            f"Missing {root}. Download orvile/rim-one-retinal-dataset-for-assessing-glaucoma"
        )

    rows = []
    for folder, split_name in [("training_set", "train"), ("test_set", "test")]:
        for label_name, label in [("glaucoma", 1), ("normal", 0)]:
            base = root / folder / label_name
            if not base.exists():
                continue
            for path in sorted(base.glob("*.png")):
                rows.append(
                    {
                        "path": str(path.resolve()),
                        "label": label,
                        "label_name": label_name,
                        "split": split_name,
                        "dataset": "rim_one",
                        "modality": "fundus",
                        "partition": "by_hospital",
                    }
                )
    if not rows:
        raise RuntimeError(f"No RIM-ONE images under {root}")

    catalog = pd.DataFrame(rows)
    print(
        f"RIM-ONE DL (by-hospital): train={sum(catalog.split=='train')}, "
        f"test={sum(catalog.split=='test')}"
    )
    return _write_split(catalog, "rim_one")


def build_biomisa_od(seed: int = 42) -> dict:
    """BIOMISA / Mendeley trghs22fpg — OD Color fundus, patient-level split.

    Glaucoma vs Healthy under Dataset/OD. Uses Color_* frames (not OCT/red-free).
    """
    from sklearn.model_selection import train_test_split

    root = RAW / "biomisa" / "Dataset" / "Dataset" / "OD"
    if not root.exists():
        raise FileNotFoundError(
            f"Missing {root}. Extract Mendeley Dataset.zip under data/raw/glaucoma/biomisa/"
        )

    # Collect per-patient Color fundus
    patient_rows: list[dict] = []
    for label_name, label in [("Glaucoma", 1), ("Healthy", 0)]:
        lab_dir = root / label_name
        if not lab_dir.exists():
            continue
        for patient in sorted(lab_dir.iterdir()):
            if not patient.is_dir():
                continue
            color_paths = []
            for eye in patient.iterdir():
                if not eye.is_dir():
                    continue
                for path in eye.iterdir():
                    if not path.is_file():
                        continue
                    if "color" in path.name.lower() and path.suffix.lower() in {
                        ".jpg",
                        ".jpeg",
                        ".png",
                    }:
                        color_paths.append(path)
            if not color_paths:
                continue
            # One representative Color frame per eye folder (first sorted)
            for path in sorted(color_paths):
                patient_rows.append(
                    {
                        "path": str(path.resolve()),
                        "label": label,
                        "label_name": "glaucoma" if label else "normal",
                        "split": "unknown",
                        "dataset": "biomisa_od",
                        "modality": "fundus",
                        "patient_id": f"{label_name}_{patient.name}",
                    }
                )

    if not patient_rows:
        raise RuntimeError(f"No BIOMISA OD Color fundus under {root}")

    catalog = pd.DataFrame(patient_rows)
    # Patient-level split to avoid eye leakage
    patients = catalog[["patient_id", "label"]].drop_duplicates()
    train_pids, test_pids = train_test_split(
        patients["patient_id"].tolist(),
        test_size=0.25,
        random_state=seed,
        stratify=patients["label"].tolist(),
    )
    train_set = set(train_pids)
    catalog["split"] = catalog["patient_id"].map(
        lambda p: "train" if p in train_set else "test"
    )
    print(
        f"BIOMISA OD Color: {len(catalog)} frames / {len(patients)} patients "
        f"({int(catalog.label.sum())} glaucoma frames)"
    )
    return _write_split(catalog, "biomisa_od")


def build_drishti(seed: int = 42) -> dict:
    """Drishti-GS1: 101 fundus images with binary glaucoma/normal folder labels.

    Uses Training/Test Images/{GLAUCOMA|NORMAL|glaucoma|normal} when present
    (Kaggle ONH mirror ships this layout).
    """
    from sklearn.model_selection import train_test_split

    root = RAW / "drishti"
    if not root.exists():
        raise FileNotFoundError(f"Missing {root}")

    rows = []
    # Only trust class folder names under .../Images/{GLAUCOMA|NORMAL}/
    # (do NOT scan full path — raw root is data/raw/glaucoma/...).
    for images_dir in root.rglob("Images"):
        if not images_dir.is_dir():
            continue
        parts_l = [p.lower() for p in images_dir.parts]
        if "training" in parts_l:
            split = "train"
        elif "test" in parts_l:
            split = "test"
        else:
            split = "unknown"
        for class_dir in images_dir.iterdir():
            if not class_dir.is_dir():
                continue
            cname = class_dir.name.lower()
            if cname == "glaucoma":
                label = 1
            elif cname in ("normal", "healthy"):
                label = 0
            else:
                continue
            for path in sorted(class_dir.glob("*")):
                if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                rows.append(
                    {
                        "path": str(path.resolve()),
                        "label": label,
                        "label_name": "glaucoma" if label else "normal",
                        "split": split,
                        "dataset": "drishti",
                        "modality": "fundus",
                    }
                )

    if not rows:
        raise RuntimeError(
            "Drishti images found but no binary glaucoma labels. "
            "Expect Images/glaucoma|normal (or GLAUCOMA|NORMAL) folders."
        )

    catalog = pd.DataFrame(rows)
    if (catalog["split"] == "unknown").any() or (catalog["split"] == "test").sum() == 0:
        idx = catalog.index.tolist()
        lbls = catalog["label"].tolist()
        train_idx, test_idx = train_test_split(
            idx, test_size=0.2, random_state=seed, stratify=lbls
        )
        catalog.loc[train_idx, "split"] = "train"
        catalog.loc[test_idx, "split"] = "test"

    print(
        f"Drishti: {len(catalog)} labeled fundus "
        f"(train={sum(catalog.split=='train')}, test={sum(catalog.split=='test')}, "
        f"{int(catalog.label.sum())} glaucoma)"
    )
    return _write_split(catalog, "drishti")


def build_harvard_combined(seed: int = 42) -> dict:
    """Merge Harvard-GD + Harvard-GDP train splits; hold out both test sets."""
    gd_train = OUT / "harvard_gd_train.csv"
    gd_test = OUT / "harvard_gd_test.csv"
    gdp_train = OUT / "harvard_gdp_train.csv"
    gdp_test = OUT / "harvard_gdp_test.csv"
    for p in (gd_train, gd_test, gdp_train, gdp_test):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run build for harvard_gd and harvard_gdp first."
            )

    train = pd.concat([pd.read_csv(gd_train), pd.read_csv(gdp_train)], ignore_index=True)
    test = pd.concat([pd.read_csv(gd_test), pd.read_csv(gdp_test)], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    train_path = OUT / "harvard_combined_train.csv"
    test_path = OUT / "harvard_combined_test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    labels = train["label"].astype(int).tolist()
    num_classes = int(train["label"].max()) + 1
    weights = _class_weights(labels, num_classes)
    STATS.mkdir(parents=True, exist_ok=True)
    stats_path = STATS / "harvard_combined_class_weights.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({"weights": weights, "counts": dict(Counter(labels))}, f, indent=2)

    print(
        f"Harvard-combined: {len(train)} train ({sum(labels)} glaucoma), "
        f"{len(test)} test"
    )
    return {
        "train": len(train),
        "test": len(test),
        "train_csv": str(train_path.relative_to(ROOT)),
        "test_csv": str(test_path.relative_to(ROOT)),
        "class_weights": str(stats_path.relative_to(ROOT)),
    }


BUILDERS = {
    "harvard_gd": build_harvard_gd,
    "harvard_gdp": build_harvard_gdp,
    "harvard_gf": build_harvard_gf,
    "harvard_combined": build_harvard_combined,
    "airogs_light": build_airogs_light,
    "gleam": build_gleam_paired,
    "gamma": build_gamma_paired,
    "acrima": build_acrima,
    "beh": build_beh,
    "rim_one": build_rim_one,
    "biomisa_od": build_biomisa_od,
    "drishti": build_drishti,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="harvard_gd", choices=list(BUILDERS))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    info = BUILDERS[args.dataset](seed=args.seed)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
