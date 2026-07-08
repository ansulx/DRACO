#!/usr/bin/env python3
"""Explore downloaded OCT datasets: labels, diagnoses, image stats."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import yaml
from PIL import Image

from paths import MMRDR_DIR, OEFI_DIR, PROCESSED_DIR, REGISTRY_PATH

MMRDR_DME_LABELS = {
    0: "No DME",
    1: "NCI DME (non-center-involving)",
    2: "CI DME (center-involving)",
}

OEFI_DR_LABELS = {
    "0": "No DR",
    0: "No DR",
    "NPDR": "NPDR",
    "PDR": "PDR",
    "-": "Unknown",
}


def _split_from_name(name: str) -> str:
    stem = Path(str(name)).stem
    if stem.startswith("tr"):
        return "train"
    if stem.startswith("ts"):
        return "test"
    return "unknown"


def _image_stats(paths: list[Path], sample_limit: int = 200) -> dict:
    if not paths:
        return {"count": 0}

    widths, heights, modes = [], [], Counter()
    missing = 0
    sample = paths[:sample_limit] if len(paths) > sample_limit else paths

    for path in sample:
        try:
            with Image.open(path) as img:
                widths.append(img.width)
                heights.append(img.height)
                modes[img.mode] += 1
        except Exception:
            missing += 1

    def _range(vals: list[int]) -> dict | None:
        if not vals:
            return None
        return {"min": min(vals), "max": max(vals), "mean": round(sum(vals) / len(vals), 1)}

    return {
        "images_on_disk": len(paths),
        "sampled": len(sample),
        "unreadable_in_sample": missing,
        "width": _range(widths),
        "height": _range(heights),
        "modes": dict(modes),
    }


def _find_images(root: Path, extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> list[Path]:
    files = []
    for ext in extensions:
        files.extend(root.rglob(f"*{ext}"))
        files.extend(root.rglob(f"*{ext.upper()}"))
    return sorted(set(files))


def explore_mmrdr() -> dict:
    oct_root = MMRDR_DIR / "MMRDR-OCT"
    csv_path = oct_root / "OCT.csv"

    result = {
        "dataset": "mmrdr",
        "path": str(MMRDR_DIR),
        "found": False,
        "errors": [],
    }

    if not csv_path.exists():
        result["errors"].append(f"Missing {csv_path}. Run: python scripts/download_datasets.py --dataset mmrdr")
        return result

    df = pd.read_csv(csv_path)
    result["found"] = True
    result["csv_columns"] = list(df.columns)
    result["rows_in_csv"] = len(df)

    image_col = next((c for c in df.columns if c.lower() in ("image", "filename", "file")), None)
    if image_col is None:
        result["errors"].append(f"No image column in OCT.csv. Columns: {list(df.columns)}")
        return result

    img_dir = oct_root / "img"
    grade_col = "grade" if "grade" in df.columns else None

    records = []
    missing_images = []

    for _, row in df.iterrows():
        rel = str(row[image_col])
        img_path = img_dir / rel if not Path(rel).is_absolute() else Path(rel)
        if not img_path.exists():
            img_path = oct_root / rel
        if not img_path.exists():
            missing_images.append(rel)

        grade = int(row[grade_col]) if grade_col and pd.notna(row[grade_col]) else None
        records.append(
            {
                "image": rel,
                "split": _split_from_name(rel),
                "grade": grade,
                "grade_label": MMRDR_DME_LABELS.get(grade, "unknown") if grade is not None else None,
                "lr": row.get("lr"),
            }
        )

    rec_df = pd.DataFrame(records)

    grade_counts = Counter(rec_df["grade_label"].dropna())
    split_counts = Counter(rec_df["split"])
    split_grade = (
        rec_df.groupby(["split", "grade_label"]).size().reset_index(name="count").to_dict("records")
        if not rec_df.empty
        else []
    )

    on_disk = _find_images(img_dir) if img_dir.exists() else _find_images(oct_root)

    result.update(
        {
            "modality": "OCT",
            "annotation_file": str(csv_path),
            "diagnosis_schema": "dme_3class",
            "diagnosis_labels": MMRDR_DME_LABELS,
            "grade_distribution": dict(sorted(grade_counts.items())),
            "split_distribution": dict(split_counts),
            "grade_by_split": split_grade,
            "missing_images": len(missing_images),
            "missing_image_samples": missing_images[:10],
            "laterality": dict(Counter(rec_df["lr"].dropna().astype(str))),
            "image_stats": _image_stats(on_disk),
            "potential_tasks": [
                "dme_3class_classification (grade 0/1/2)",
                "dme_binary (grade 0 vs 1+2)",
                "ci_dme_detection (grade 2 vs rest)",
            ],
        }
    )
    return result


def explore_oefi() -> dict:
    csv_path = OEFI_DIR / "OCT.csv"
    oct_root = OEFI_DIR / "OCT"

    result = {
        "dataset": "oefi",
        "path": str(OEFI_DIR),
        "found": False,
        "errors": [],
    }

    if not csv_path.exists():
        result["errors"].append(f"Missing {csv_path}. Run: python scripts/download_datasets.py --dataset oefi")
        return result

    df = pd.read_csv(csv_path)
    result["found"] = True
    result["csv_columns"] = list(df.columns)
    result["rows_in_csv"] = len(df)

    name_col = "Name" if "Name" in df.columns else df.columns[0]
    missing_images = []
    resolved_paths = []

    for name in df[name_col].astype(str):
        matches = list(oct_root.rglob(f"{name}.jpg")) if oct_root.exists() else []
        if not matches:
            matches = list(OEFI_DIR.rglob(f"{name}.jpg"))
        if matches:
            resolved_paths.append(matches[0])
        else:
            missing_images.append(name)

    dme_counts = Counter(df["DME"].astype(int).map({0: "No DME", 1: "DME present"}))
    dr_counts = Counter(df["DR"].map(lambda v: OEFI_DR_LABELS.get(v, str(v))))

    result.update(
        {
            "modality": "OCT",
            "annotation_file": str(csv_path),
            "diagnosis_schemas": ["dme_binary", "dr_npdr_pdr"],
            "dme_distribution": dict(dme_counts),
            "dr_distribution": dict(dr_counts),
            "missing_images": len(missing_images),
            "missing_image_samples": missing_images[:10],
            "images_resolved": len(resolved_paths),
            "expected_image_size": "1408x573 (per dataset docs)",
            "image_stats": _image_stats(resolved_paths),
            "potential_tasks": [
                "dme_binary_classification (DME 0/1)",
                "dr_coarse_classification (0 / NPDR / PDR; exclude '-')",
                "external_validation for MMRDR-trained models",
            ],
        }
    )
    return result


def print_report(report: dict) -> None:
    print("=" * 60)
    print("OCT DR — Local Dataset Explorer")
    print(f"Generated: {report['generated_at']}")
    print("=" * 60)

    for key in ("mmrdr", "oefi"):
        ds = report["datasets"].get(key, {})
        print(f"\n## {key.upper()}")
        print(f"Path: {ds.get('path', 'n/a')}")

        if not ds.get("found"):
            for err in ds.get("errors", ["Not found"]):
                print(f"  ! {err}")
            continue

        print(f"CSV rows: {ds.get('rows_in_csv')}")
        print(f"Columns: {ds.get('csv_columns')}")

        if key == "mmrdr":
            print("\nDiagnosis (DME 3-class on OCT):")
            for label, count in ds.get("grade_distribution", {}).items():
                print(f"  {label}: {count}")
            print("\nTrain / test split:")
            for split, count in ds.get("split_distribution", {}).items():
                print(f"  {split}: {count}")
            print("\nGrade by split:")
            for row in ds.get("grade_by_split", []):
                print(f"  {row['split']} | {row['grade_label']}: {row['count']}")
        else:
            print("\nDME (binary):")
            for label, count in ds.get("dme_distribution", {}).items():
                print(f"  {label}: {count}")
            print("\nDR (coarse on OCT):")
            for label, count in ds.get("dr_distribution", {}).items():
                print(f"  {label}: {count}")

        stats = ds.get("image_stats", {})
        if stats.get("width"):
            w, h = stats["width"], stats["height"]
            print(
                f"\nImages: {stats.get('images_on_disk', stats.get('sampled', 0))} on disk | "
                f"size ~{w['min']}-{w['max']} x {h['min']}-{h['max']} px"
            )
        if ds.get("missing_images"):
            print(f"Missing images: {ds['missing_images']}")
            if ds.get("missing_image_samples"):
                print(f"  samples: {ds['missing_image_samples']}")

        print("\nPotential tasks:")
        for task in ds.get("potential_tasks", []):
            print(f"  - {task}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore local MMRDR and OEFI data")
    parser.add_argument(
        "--dataset",
        choices=["mmrdr", "oefi", "all"],
        default="all",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=PROCESSED_DIR / "local_audit.json",
        help="Write JSON report to this path",
    )
    parser.add_argument("--no-save", action="store_true", help="Skip writing JSON report")
    args = parser.parse_args()

    datasets = {}
    if args.dataset in ("mmrdr", "all"):
        datasets["mmrdr"] = explore_mmrdr()
    if args.dataset in ("oefi", "all"):
        datasets["oefi"] = explore_oefi()

    registry = {}
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, encoding="utf-8") as handle:
            registry = yaml.safe_load(handle)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "registry": str(REGISTRY_PATH),
        "recommended_primary_task": registry.get("recommended_task_order", [{}])[0],
        "datasets": datasets,
    }

    print_report(report)

    if not args.no_save:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        print(f"\nReport saved: {args.json}")


if __name__ == "__main__":
    main()
