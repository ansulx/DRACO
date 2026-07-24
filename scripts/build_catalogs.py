#!/usr/bin/env python3
"""Build processed catalog CSVs for MMRDR and OEFI."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import MMRDR_DIR, OEFI_DIR, PROCESSED_DIR, ROOT

DME_NAMES = {0: "no_dme", 1: "nci_dme", 2: "ci_dme"}


def _split_from_name(name: str) -> str:
    stem = Path(str(name)).stem
    if stem.startswith("tr"):
        return "train"
    if stem.startswith("ts"):
        return "test"
    return "unknown"


def build_mmrdr(out_dir: Path) -> dict:
    oct_root = MMRDR_DIR / "MMRDR-OCT"
    csv_path = oct_root / "OCT.csv"
    img_dir = oct_root / "img"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {csv_path}. Run: python scripts/extract_mmrdr_oct.py"
        )

    df = pd.read_csv(csv_path)
    image_col = next(
        (c for c in df.columns if c.lower() in ("image", "filename", "file", "path")),
        None,
    )
    if image_col is None:
        raise ValueError(f"No image column in OCT.csv: {list(df.columns)}")
    if "grade" not in df.columns:
        raise ValueError(f"No grade column in OCT.csv: {list(df.columns)}")

    rows = []
    missing = 0
    for _, row in df.iterrows():
        rel = str(row[image_col]).replace("\\", "/")
        # Paths in CSV may be like img/tr000001.jpg or just tr000001.jpg
        candidates = [
            img_dir / Path(rel).name,
            oct_root / rel,
            img_dir / rel,
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            missing += 1
            continue
        grade = int(row["grade"])
        split = _split_from_name(Path(rel).name)
        rows.append(
            {
                "path": str(path.resolve()),
                "label": grade,
                "label_name": DME_NAMES.get(grade, str(grade)),
                "split": split,
                "dataset": "mmrdr",
                "eye": row.get("lr", ""),
            }
        )

    catalog = pd.DataFrame(rows)
    train = catalog[catalog["split"] == "train"]
    test = catalog[catalog["split"] == "test"]
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "mmrdr_train.csv"
    test_path = out_dir / "mmrdr_test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    counts = Counter(train["label"].tolist())
    total = sum(counts.values()) or 1
    # inverse frequency class weights, normalized
    weights = {c: total / (len(counts) * counts[c]) for c in sorted(counts)}
    weight_list = [weights[c] for c in sorted(weights)]

    stats = {
        "n_train": len(train),
        "n_test": len(test),
        "missing_images": missing,
        "train_label_counts": {DME_NAMES[k]: int(v) for k, v in sorted(counts.items())},
        "class_weights": weight_list,
        "class_weight_map": {DME_NAMES[k]: float(v) for k, v in sorted(weights.items())},
        "columns": list(df.columns),
    }
    return stats


def build_oefi(out_dir: Path) -> dict:
    csv_path = OEFI_DIR / "OCT.csv"
    oct_root = OEFI_DIR / "OCT"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)
    name_col = "Name" if "Name" in df.columns else df.columns[0]
    rows = []
    missing = 0
    for _, row in df.iterrows():
        name = str(row[name_col])
        matches = list(oct_root.rglob(f"{name}.jpg")) if oct_root.exists() else []
        if not matches:
            missing += 1
            continue
        dme = int(row["DME"])
        rows.append(
            {
                "path": str(matches[0].resolve()),
                "label": dme,
                "label_name": "dme" if dme == 1 else "no_dme",
                "split": "external",
                "dataset": "oefi",
                "dr": row.get("DR", ""),
            }
        )

    catalog = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "oefi_external.csv"
    catalog.to_csv(out_path, index=False)
    return {
        "n_external": len(catalog),
        "missing_images": missing,
        "dme_counts": dict(Counter(catalog["label_name"].tolist())),
    }


def build_octid(out_dir: Path) -> dict:
    base = ROOT / "data" / "raw" / "octid"
    rows = []
    for sub, label, name in [("dr", 1, "dr"), ("normal", 0, "normal")]:
        folder = base / sub
        if not folder.exists():
            continue
        for path in list(folder.glob("*.jpeg")) + list(folder.glob("*.jpg")):
            rows.append(
                {
                    "path": str(path.resolve()),
                    "label": label,
                    "label_name": name,
                    "split": "supplement",
                    "dataset": "octid",
                }
            )
    if not rows:
        return {"n": 0, "skipped": True}
    catalog = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(out_dir / "octid_supplement.csv", index=False)
    return {"n": len(catalog), "counts": dict(Counter(catalog["label_name"].tolist()))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DRACO processed catalogs")
    parser.add_argument("--skip-mmrdr", action="store_true")
    parser.add_argument("--skip-oefi", action="store_true")
    parser.add_argument("--skip-octid", action="store_true")
    args = parser.parse_args()

    catalogs = PROCESSED_DIR / "catalogs"
    stats_dir = PROCESSED_DIR / "stats"
    catalogs.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    all_stats: dict = {}

    if not args.skip_mmrdr:
        print("Building MMRDR catalogs...")
        all_stats["mmrdr"] = build_mmrdr(catalogs)
        print(f"  train={all_stats['mmrdr']['n_train']} test={all_stats['mmrdr']['n_test']}")
        with open(stats_dir / "class_weights.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "weights": all_stats["mmrdr"]["class_weights"],
                    "map": all_stats["mmrdr"]["class_weight_map"],
                    "label_names": DME_NAMES,
                },
                f,
                indent=2,
            )

    if not args.skip_oefi:
        print("Building OEFI external catalog...")
        all_stats["oefi"] = build_oefi(catalogs)
        print(f"  external={all_stats['oefi']['n_external']}")

    if not args.skip_octid:
        print("Building OCTID supplement catalog...")
        all_stats["octid"] = build_octid(catalogs)
        print(f"  octid={all_stats['octid']}")

    with open(stats_dir / "image_stats.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)

    print(f"Wrote catalogs to {catalogs}")
    print(f"Wrote stats to {stats_dir}")


if __name__ == "__main__":
    main()
