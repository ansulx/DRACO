#!/usr/bin/env python3
"""Download MMRDR and OEFI datasets into data/raw/."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# Allow `python scripts/download_datasets.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests
from tqdm import tqdm

from paths import FIGSHARE_MMRDR_ARTICLE, MMRDR_DIR, OEFI_DIR, RAW_DIR

OEFI_REPO = "https://github.com/Traslational-Visual-Health-Laboratory/OCT-AND-EYE-FUNDUS-DATASET.git"
FIGSHARE_API = f"https://api.figshare.com/v2/articles/{FIGSHARE_MMRDR_ARTICLE}"


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n} B"


def _download_url(url: str, dest: Path, expected_size: int | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "oct-dr-benchmark/1.0"}

    resume_at = 0
    if dest.exists():
        resume_at = dest.stat().st_size
        if expected_size and resume_at >= expected_size:
            print(f"  skip (complete): {dest.name}")
            return
        if resume_at > 0:
            headers["Range"] = f"bytes={resume_at}-"

    with requests.get(url, headers=headers, stream=True, timeout=120) as response:
        if response.status_code == 416:
            print(f"  skip (complete): {dest.name}")
            return
        response.raise_for_status()

        total = expected_size
        if total is None and "content-length" in response.headers:
            total = int(response.headers["content-length"]) + resume_at

        mode = "ab" if resume_at else "wb"
        with open(dest, mode) as handle, tqdm(
            total=total,
            initial=resume_at,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest.name,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    bar.update(len(chunk))


def _fetch_mmrdr_files() -> list[dict]:
    with urllib.request.urlopen(FIGSHARE_API, timeout=60) as response:
        article = json.load(response)
    return article["files"]


def download_mmrdr(oct_only: bool = True, keep_zip: bool = False) -> None:
    """Download MMRDR from Figshare (~18 GB split archive)."""
    files = _fetch_mmrdr_files()
    zip_parts = sorted(
        [f for f in files if re.match(r"MMRDR\.zip\.\d{3}$", f["name"])],
        key=lambda f: f["name"],
    )
    readme = next((f for f in files if f["name"] == "README.md"), None)

    if not zip_parts:
        raise RuntimeError("No MMRDR zip parts found on Figshare.")

    total_size = sum(int(f["size"]) for f in zip_parts)
    print(f"MMRDR: {len(zip_parts)} parts, total {_human_bytes(total_size)}")
    if oct_only:
        print("  --oct-only: will extract MMRDR-OCT/ only after download")

    staging = MMRDR_DIR / "_staging"
    staging.mkdir(parents=True, exist_ok=True)

    for part in zip_parts:
        dest = staging / part["name"]
        print(f"Downloading {part['name']} ({_human_bytes(int(part['size']))})")
        _download_url(part["download_url"], dest, int(part["size"]))

    merged_zip = staging / "MMRDR.zip"
    if not merged_zip.exists() or merged_zip.stat().st_size != total_size:
        print(f"Merging parts -> {merged_zip.name}")
        with open(merged_zip, "wb") as out:
            for part in zip_parts:
                part_path = staging / part["name"]
                with open(part_path, "rb") as inp:
                    shutil.copyfileobj(inp, out, length=1024 * 1024)

    if readme:
        _download_url(readme["download_url"], MMRDR_DIR / "README.md", int(readme["size"]))

    print("Extracting...")
    prefix = "MMRDR/MMRDR-OCT/" if oct_only else "MMRDR/"
    alt_prefix = "MMRDR-OCT/" if oct_only else None

    with zipfile.ZipFile(merged_zip, "r") as archive:
        members = archive.namelist()
        if oct_only:
            selected = [
                m
                for m in members
                if m.startswith(prefix) or (alt_prefix and m.startswith(alt_prefix))
            ]
            if not selected:
                raise RuntimeError(
                    f"No paths matching {prefix!r} in archive. "
                    f"Sample entries: {members[:5]}"
                )
        else:
            selected = [m for m in members if not m.endswith("/")]

        for member in tqdm(selected, desc="extract", unit="file"):
            archive.extract(member, MMRDR_DIR)

    # Normalize layout: ensure MMRDR-OCT lives directly under mmrdr/
    for candidate in (MMRDR_DIR / "MMRDR" / "MMRDR-OCT", MMRDR_DIR / "MMRDR-OCT"):
        if candidate.exists() and candidate.is_dir():
            target = MMRDR_DIR / "MMRDR-OCT"
            if candidate != target and not target.exists():
                shutil.move(str(candidate), str(target))
            break

    nested = MMRDR_DIR / "MMRDR"
    if nested.exists() and nested.is_dir() and not oct_only:
        for child in nested.iterdir():
            dest = MMRDR_DIR / child.name
            if not dest.exists():
                shutil.move(str(child), str(dest))
        nested.rmdir()

    if not keep_zip:
        shutil.rmtree(staging, ignore_errors=True)
    else:
        print(f"Kept merged zip at {merged_zip}")

    oct_csv = MMRDR_DIR / "MMRDR-OCT" / "OCT.csv"
    if oct_only and not oct_csv.exists():
        raise RuntimeError(f"Extraction finished but {oct_csv} is missing.")

    print(f"MMRDR ready at {MMRDR_DIR}")


def download_oefi(oct_only: bool = True) -> None:
    """Clone OEFI from GitHub."""
    if not shutil.which("git"):
        raise RuntimeError("git is required to download OEFI. Install Git and retry.")

    OEFI_DIR.mkdir(parents=True, exist_ok=True)
    keep = {"OCT", "OCT.csv", "README.md"}
    if not oct_only:
        keep.update({"EYE FUNDUS", "EYE FUNDUS.csv"})

    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}

    with tempfile.TemporaryDirectory(prefix="oefi_clone_") as tmp:
        clone_dir = Path(tmp) / "repo"
        print("Cloning OEFI repository (shallow)...")
        subprocess.run(
            ["git", "clone", "--depth", "1", OEFI_REPO, str(clone_dir)],
            check=True,
            env=env,
        )

        for name in keep:
            src = clone_dir / name
            if not src.exists():
                raise RuntimeError(f"Expected {name} in OEFI repository")

            dest = OEFI_DIR / name
            if dest.exists():
                shutil.rmtree(dest) if dest.is_dir() else dest.unlink()

            if src.is_dir():
                shutil.copytree(src, dest)
            else:
                shutil.copy2(src, dest)

    print(f"OEFI ready at {OEFI_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OCT DR datasets")
    parser.add_argument(
        "--dataset",
        choices=["mmrdr", "oefi", "all"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    parser.add_argument(
        "--oct-only",
        action="store_true",
        default=True,
        help="Download OCT subset only (default: true)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Download all modalities (disables --oct-only)",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep merged MMRDR zip after extraction",
    )
    args = parser.parse_args()
    oct_only = not args.full

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        if args.dataset in ("mmrdr", "all"):
            download_mmrdr(oct_only=oct_only, keep_zip=args.keep_zip)
        if args.dataset in ("oefi", "all"):
            download_oefi(oct_only=oct_only)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
