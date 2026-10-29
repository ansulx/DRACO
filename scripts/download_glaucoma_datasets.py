#!/usr/bin/env python3
"""Download glaucoma datasets for GLAUCO training.

Supported (auto-download via Hugging Face):
  harvard_gd   — 500 RNFLT maps (~182 MB)
  harvard_gdp  — 1,000 patients
  harvard_gf   — 3,300 patients (~20 GB) — needs HF token + accept license on HF

Manual / large:
  airogs_light — Kaggle
  airogs_full  — Zenodo (approval required)
  gamma        — Grand Challenge registration
  gleam        — Kaggle

Usage:
  python scripts/download_glaucoma_datasets.py --dataset harvard_gd
  python scripts/download_glaucoma_datasets.py --dataset all-smoke
  HF_TOKEN=hf_... python scripts/download_glaucoma_datasets.py --dataset harvard_gd
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "glaucoma"


def _ensure_hf():
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
        return snapshot_download, hf_hub_download
    except ImportError:
        print("Install: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)


def download_harvard(repo_id: str, out_dir: Path, token: str | None) -> None:
    snapshot_download, _ = _ensure_hf()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {out_dir}")
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(out_dir),
        token=token,
    )
    print(f"Done: {path}")


def download_kaggle(dataset: str, out_dir: Path) -> None:
    try:
        import kaggle
    except ImportError:
        print(
            "Kaggle CLI not installed.\n"
            "  pip install kaggle\n"
            "  Place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME/KAGGLE_KEY\n",
            file=sys.stderr,
        )
        _print_kaggle_manual(dataset, out_dir)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Kaggle {dataset} -> {out_dir}")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=str(out_dir), unzip=True)
    print(f"Done: {out_dir}")


def _print_kaggle_manual(dataset: str, out_dir: Path) -> None:
    print(
        f"\nManual Kaggle download for {dataset}:\n"
        f"  1. https://www.kaggle.com/datasets/{dataset.replace('/', '/')}\n"
        f"  2. Extract to: {out_dir}\n"
    )


def print_manual(name: str, instructions: str) -> None:
    print(f"\n=== {name} (manual download) ===\n{instructions}\n")


MANUAL = {
    "harvard_gf": (
        "Harvard-GF (20.4 GB)\n"
        "  Zenodo: https://zenodo.org/records/13178701\n"
        "  GitHub: https://github.com/Harvard-Ophthalmology-AI-Lab/Harvard-GF\n"
        f"  Extract to: {RAW / 'harvard_gf'}"
    ),
    "airogs_full": (
        "Full AIROGS (54.9 GB, manual approval on Zenodo)\n"
        "  https://zenodo.org/records/5793241\n"
        f"  Extract to: {RAW / 'airogs_full'}"
    ),
    "gamma": (
        "GAMMA challenge (paired fundus + OCT)\n"
        "  Register: https://gamma.grand-challenge.org/\n"
        f"  Extract to: {RAW / 'gamma'}"
    ),
    "gleam": (
        "GLEAM tri-modal dataset (2026)\n"
        "  Kaggle: https://www.kaggle.com/datasets/zhangyiyinge/gleam-dataset\n"
        "  Code: https://github.com/microewing/HAMM\n"
        f"  Extract to: {RAW / 'gleam'}"
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download GLAUCO glaucoma datasets")
    parser.add_argument(
        "--dataset",
        default="harvard_gd",
        choices=[
            "harvard_gd",
            "harvard_gdp",
            "harvard_gf",
            "airogs_light",
            "airogs_full",
            "gamma",
            "gleam",
            "all-smoke",
        ],
    )
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    targets = [args.dataset]
    if args.dataset == "all-smoke":
        targets = ["harvard_gd"]

    for name in targets:
        if name == "harvard_gd":
            download_harvard(
                "harvardairobotics/Harvard-GD",
                RAW / "harvard_gd",
                args.token,
            )
        elif name == "harvard_gdp":
            download_harvard(
                "harvardairobotics/Harvard-GDP",
                RAW / "harvard_gdp",
                args.token,
            )
        elif name == "harvard_gf":
            download_harvard(
                "harvardairobotics/Harvard-GF",
                RAW / "harvard_gf",
                args.token,
            )
        elif name == "airogs_light":
            download_kaggle(
                "deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2",
                RAW / "airogs_light",
            )
        elif name == "airogs_full":
            print_manual("AIROGS Full", MANUAL["airogs_full"])
        elif name == "gamma":
            print_manual("GAMMA", MANUAL["gamma"])
        elif name == "gleam":
            print_manual("GLEAM", MANUAL["gleam"])

    print("\nNext: python scripts/build_glaucoma_catalogs.py --dataset harvard_gd")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
