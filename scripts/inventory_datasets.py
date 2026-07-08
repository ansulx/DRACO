#!/usr/bin/env python3
"""Inventory all downloaded OCT datasets — zips and extracted folders."""

from __future__ import annotations

import json
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from PIL import Image

from paths import DATA_DIR, MMRDR_DIR, OEFI_DIR, PROCESSED_DIR, RAW_DIR, ROOT

DME_3 = {0: "No DME", 1: "NCI DME", 2: "CI DME"}
OEFI_DR = {"0": "No DR", 0: "No DR", "NPDR": "NPDR", "PDR": "PDR", "-": "Unknown"}


def _gb(path: Path) -> float:
    return round(path.stat().st_size / (1024**3), 2) if path.exists() else 0


def _image_stats(paths: list[Path], limit: int = 100) -> dict:
    if not paths:
        return {}
    widths, heights = [], []
    for p in paths[:limit]:
        try:
            with Image.open(p) as im:
                widths.append(im.width)
                heights.append(im.height)
        except Exception:
            pass
    if not widths:
        return {"sampled": 0}
    return {
        "sampled": len(widths),
        "width": f"{min(widths)}–{max(widths)}",
        "height": f"{min(heights)}–{max(heights)}",
    }


def scan_root_zips() -> list[dict]:
    mapping = {
        "29423747.zip": {
            "dataset": "MMRDR",
            "description": "Figshare bundle — 9 split parts + README (needs merge + extract)",
        },
        "OCT-AND-EYE-FUNDUS-DATASET-main.zip": {
            "dataset": "OEFI",
            "description": "Full OEFI zip (OCT + fundus)",
        },
        "doi-10.5683-sp-flgzze.zip": {
            "dataset": "OCTID-DR",
            "description": "OCTID diabetic retinopathy images",
        },
        "doi-10.5683-sp-wlw4zt.zip": {
            "dataset": "OCTID-Normal",
            "description": "OCTID normal retina images",
        },
        "rscbjbr9sj-3.zip": {
            "dataset": "Kermany",
            "description": "Cell 2018 OCT + chest X-ray (nested ZhangLabData.zip)",
        },
    }
    results = []
    for name, meta in mapping.items():
        path = ROOT / name
        entry = {
            "file": name,
            "dataset": meta["dataset"],
            "description": meta["description"],
            "on_disk": path.exists(),
            "size_gb": _gb(path),
            "extracted_to": None,
            "status": "missing",
        }
        if not path.exists():
            results.append(entry)
            continue

        entry["status"] = "zip_only"
        if meta["dataset"] == "MMRDR":
            with zipfile.ZipFile(path) as zf:
                entry["zip_contents"] = zf.namelist()
                entry["note"] = "Extract MMRDR-OCT after merging MMRDR.zip.001–009"
        elif meta["dataset"] == "Kermany":
            entry["note"] = "Contains CellData/OCT/ (~109k B-scans) + chest X-ray — extract OCT only"
        elif meta["dataset"] == "OEFI":
            with zipfile.ZipFile(path) as zf:
                oct_n = sum(1 for n in zf.namelist() if "/OCT/" in n and n.lower().endswith(".jpg"))
                fundus_n = sum(1 for n in zf.namelist() if "EYE FUNDUS/" in n and n.lower().endswith(".jpg"))
                entry["oct_images_in_zip"] = oct_n
                entry["fundus_images_in_zip"] = fundus_n

        # check if extracted
        if meta["dataset"] == "MMRDR" and (MMRDR_DIR / "MMRDR-OCT" / "OCT.csv").exists():
            entry["status"] = "extracted"
            entry["extracted_to"] = str(MMRDR_DIR)
        elif meta["dataset"] == "OEFI" and (OEFI_DIR / "OCT.csv").exists():
            entry["status"] = "extracted"
            entry["extracted_to"] = str(OEFI_DIR)
        elif meta["dataset"].startswith("OCTID") and (RAW_DIR / "octid").exists():
            sub = "dr" if "DR" in meta["dataset"] else "normal"
            if list((RAW_DIR / "octid" / sub).glob("*")):
                entry["status"] = "extracted"
                entry["extracted_to"] = str(RAW_DIR / "octid" / sub)

        results.append(entry)
    return results


def inventory_oefi() -> dict:
    csv_path = OEFI_DIR / "OCT.csv"
    oct_root = OEFI_DIR / "OCT"
    out = {"name": "OEFI", "modality": "OCT", "ready": False}
    if not csv_path.exists():
        out["error"] = "Not extracted — unzip OCT-AND-EYE-FUNDUS-DATASET-main.zip"
        return out

    df = pd.read_csv(csv_path)
    images = list(oct_root.rglob("*.jpg")) if oct_root.exists() else []
    out.update(
        {
            "ready": True,
            "path": str(OEFI_DIR),
            "images": len(images),
            "csv_rows": len(df),
            "labels": {
                "DME_binary": dict(Counter(df["DME"].map({0: "No DME", 1: "DME"}))),
                "DR_coarse": dict(Counter(df["DR"].map(lambda v: OEFI_DR.get(v, str(v))))),
            },
            "diagnosis_type": "Binary DME + coarse DR (0/NPDR/PDR/unknown)",
            "image_stats": _image_stats(images),
            "use_for": "External validation (Mexico, multi-site)",
            "skip": "EYE FUNDUS/ folder — separate fundus pipeline",
        }
    )
    return out


def inventory_mmrdr() -> dict:
    csv_path = MMRDR_DIR / "MMRDR-OCT" / "OCT.csv"
    img_dir = MMRDR_DIR / "MMRDR-OCT" / "img"
    out = {"name": "MMRDR", "modality": "OCT", "ready": False}
    if not csv_path.exists():
        out["error"] = "Not extracted - merge MMRDR.zip.001-009 from 29423747.zip, then unzip"
        out["zip_location"] = str(ROOT / "29423747.zip")
        out["expected"] = {
            "oct_images": 2938,
            "labels": "3-class DME (0=No, 1=NCI, 2=CI)",
        }
        return out

    df = pd.read_csv(csv_path)
    image_col = "image" if "image" in df.columns else df.columns[0]
    images = list(img_dir.rglob("*.jpg")) if img_dir.exists() else []
    grades = Counter(int(g) for g in df["grade"] if pd.notna(g))
    splits = Counter()
    for img in df[image_col].astype(str):
        splits["train" if Path(img).stem.startswith("tr") else "test" if Path(img).stem.startswith("ts") else "other"] += 1

    out.update(
        {
            "ready": True,
            "path": str(MMRDR_DIR / "MMRDR-OCT"),
            "images": len(images),
            "csv_rows": len(df),
            "labels": {"DME_3class": {DME_3.get(k, k): v for k, v in sorted(grades.items())}},
            "split": dict(splits),
            "diagnosis_type": "3-class DME staging on OCT",
            "image_stats": _image_stats(images),
            "use_for": "Primary train/benchmark (2026 SOTA reference)",
        }
    )
    return out


def inventory_octid() -> dict:
    base = RAW_DIR / "octid"
    out = {"name": "OCTID", "modality": "OCT", "ready": False, "subsets": {}}
    for sub, label in [("dr", "Diabetic Retinopathy"), ("normal", "Normal")]:
        folder = base / sub
        if not folder.exists():
            continue
        files = list(folder.glob("*.jpeg")) + list(folder.glob("*.jpg"))
        out["subsets"][sub] = {
            "label": label,
            "images": len(files),
            "image_stats": _image_stats(files),
        }
    if out["subsets"]:
        out["ready"] = True
        out["path"] = str(base)
        out["diagnosis_type"] = "Disease category only (no DME grade)"
        out["labels"] = {"DR": out["subsets"].get("dr", {}).get("images", 0), "Normal": out["subsets"].get("normal", {}).get("images", 0)}
        out["use_for"] = "Supplementary DR vs normal / RETFound-style benchmark"
    else:
        out["error"] = "Extract doi-10.5683-sp-*.zip to data/raw/octid/"
    return out


def inventory_kermany() -> dict:
    oct_root = RAW_DIR / "kermany" / "CellData" / "OCT"
    out = {
        "name": "Kermany (Cell 2018)",
        "modality": "OCT",
        "ready": False,
        "zip_location": str(ROOT / "rscbjbr9sj-3.zip"),
    }
    if oct_root.exists():
        counts = {}
        images = []
        for split in ("train", "test"):
            for cls in ("CNV", "DME", "DRUSEN", "NORMAL"):
                folder = oct_root / split / cls
                if folder.exists():
                    files = list(folder.glob("*.jpeg")) + list(folder.glob("*.jpg"))
                    counts[f"{split}/{cls}"] = len(files)
                    images.extend(files)
        out.update(
            {
                "ready": True,
                "path": str(oct_root),
                "total_images": len(images),
                "class_counts": counts,
                "diagnosis_type": "4-class maculopathy (CNV, DME, DRUSEN, NORMAL)",
                "image_stats": _image_stats(images),
                "use_for": "Large-scale DME slices; urgent-referral (CNV+DME vs rest)",
                "skip": "CellData/chest_xray/ — not retina",
            }
        )
    else:
        out["error"] = "Not extracted - unzip rscbjbr9sj-3.zip then ZhangLabData.zip; use CellData/OCT/ only"
        out["expected_in_zip"] = {
            "total_oct_images": 109309,
            "train": {"CNV": 37205, "DME": 11348, "DRUSEN": 8616, "NORMAL": 51140},
            "test": {"CNV": 250, "DME": 250, "DRUSEN": 250, "NORMAL": 250},
        }
    return out


def print_summary(report: dict) -> None:
    print("=" * 70)
    print("DOWNLOADED DATA INVENTORY")
    print("=" * 70)

    print("\n## Zip files in project root\n")
    for z in report["root_zips"]:
        status = z["status"].upper()
        size = f"{z.get('size_gb', 0)} GB" if z.get("size_gb") else "—"
        print(f"  [{status:12}] {z['dataset']:12} {z['file']} ({size})")
        if z.get("note"):
            print(f"               -> {z['note']}")

    print("\n## Dataset details (OCT-only focus)\n")
    for key in ("mmrdr", "oefi", "octid", "kermany"):
        ds = report["datasets"][key]
        print(f"### {ds['name']}")
        print(f"  Ready: {'YES' if ds.get('ready') else 'NO'}")
        if ds.get("error"):
            print(f"  Action: {ds['error']}")
        if ds.get("path"):
            print(f"  Path: {ds['path']}")
        if ds.get("images"):
            print(f"  Images: {ds['images']}")
        elif ds.get("total_images"):
            print(f"  Images: {ds['total_images']}")
        if ds.get("csv_rows"):
            print(f"  CSV rows: {ds['csv_rows']}")
        if ds.get("labels"):
            print(f"  Labels: {ds['labels']}")
        if ds.get("split"):
            print(f"  Split: {ds['split']}")
        if ds.get("class_counts"):
            print("  Class counts:")
            for k, v in sorted(ds["class_counts"].items()):
                print(f"    {k}: {v}")
        if ds.get("subsets"):
            for sub, info in ds["subsets"].items():
                print(f"  {sub}: {info['images']} images ({info['label']})")
        if ds.get("diagnosis_type"):
            print(f"  Diagnosis: {ds['diagnosis_type']}")
        if ds.get("use_for"):
            print(f"  Use for: {ds['use_for']}")
        if ds.get("skip"):
            print(f"  Skip: {ds['skip']}")
        print()

    print("## Recommended roles\n")
    for line in report["recommended_roles"]:
        print(f"  - {line}")


def main() -> None:
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root_zips": scan_root_zips(),
        "datasets": {
            "mmrdr": inventory_mmrdr(),
            "oefi": inventory_oefi(),
            "octid": inventory_octid(),
            "kermany": inventory_kermany(),
        },
        "recommended_roles": [
            "MMRDR-OCT -> primary training (3-class DME) - NEEDS EXTRACT",
            "OEFI-OCT -> external validation - READY",
            "OCTID (DR+Normal) -> supplementary DR vs normal - READY",
            "Kermany DME class -> optional scale (~11k train DME) - NEEDS EXTRACT",
            "Ignore: OEFI fundus, Kermany chest X-ray, MMRDR CFP/UWF (separate fundus pipeline)",
        ],
    }

    ready_oct = 0
    ds = report["datasets"]
    if ds["oefi"].get("ready"):
        ready_oct += ds["oefi"].get("images", 0)
    if ds["mmrdr"].get("ready"):
        ready_oct += ds["mmrdr"].get("images", 0)
    if ds["octid"].get("ready"):
        ready_oct += sum(s.get("images", 0) for s in ds["octid"].get("subsets", {}).values())
    report["total_oct_images_ready"] = ready_oct

    print_summary(report)

    out_path = PROCESSED_DIR / "inventory.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Full report: {out_path}")


if __name__ == "__main__":
    main()
