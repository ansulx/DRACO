#!/usr/bin/env python3
"""Extract MMRDR-OCT only from the Figshare multi-part archive bundle."""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import MMRDR_DIR, ROOT
from tqdm import tqdm

FIGSHARE_BUNDLE = ROOT / "29423747.zip"
STAGING = MMRDR_DIR / "_staging"
MERGED_ZIP = STAGING / "MMRDR.zip"
PART_GLOB = "MMRDR.zip."


def extract_parts_from_bundle(bundle: Path, staging: Path) -> list[Path]:
    staging.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    with zipfile.ZipFile(bundle, "r") as zf:
        names = sorted(n for n in zf.namelist() if n.startswith(PART_GLOB) or n.endswith(".zip.001"))
        # Prefer exact multi-part names
        names = sorted(n for n in zf.namelist() if n.startswith("MMRDR.zip."))
        if not names:
            raise RuntimeError(f"No MMRDR.zip.* parts found in {bundle}")
        print(f"Found {len(names)} parts in {bundle.name}")
        for name in names:
            dest = staging / Path(name).name
            if dest.exists() and dest.stat().st_size == zf.getinfo(name).file_size:
                print(f"  skip (exists): {dest.name}")
            else:
                print(f"  extracting {name} ...")
                with zf.open(name) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out, length=1024 * 1024)
            parts.append(dest)
            if name == "README.md":
                continue
        # also copy README if present
        if "README.md" in zf.namelist():
            readme_dest = MMRDR_DIR / "README.md"
            with zf.open("README.md") as src, open(readme_dest, "wb") as out:
                shutil.copyfileobj(src, out)
    parts = sorted(p for p in staging.glob("MMRDR.zip.*"))
    return parts


def merge_parts(parts: list[Path], dest: Path) -> Path:
    expected = sum(p.stat().st_size for p in parts)
    if dest.exists() and dest.stat().st_size == expected:
        print(f"Merged zip already present ({expected / 1e9:.2f} GB)")
        return dest
    print(f"Merging {len(parts)} parts -> {dest.name} ({expected / 1e9:.2f} GB)")
    with open(dest, "wb") as out:
        for part in parts:
            with open(part, "rb") as inp:
                while True:
                    chunk = inp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
    return dest


def extract_oct_only(merged: Path, out_dir: Path) -> Path:
    """Extract MMRDR-OCT subtree into data/raw/mmrdr/MMRDR-OCT."""
    target = out_dir / "MMRDR-OCT"
    csv_path = target / "OCT.csv"
    if csv_path.exists() and (target / "img").exists():
        n_img = len(list((target / "img").glob("*.jpg")))
        if n_img > 0:
            print(f"MMRDR-OCT already extracted ({n_img} images) at {target}")
            return target

    prefixes = ("MMRDR/MMRDR-OCT/", "MMRDR-OCT/")
    print(f"Extracting OCT subset from {merged.name} ...")
    with zipfile.ZipFile(merged, "r") as archive:
        members = [
            m
            for m in archive.namelist()
            if any(m.startswith(p) for p in prefixes) and not m.endswith("/")
        ]
        if not members:
            sample = archive.namelist()[:10]
            raise RuntimeError(f"No MMRDR-OCT members found. Sample: {sample}")

        for member in tqdm(members, desc="extract OCT", unit="file"):
            # Normalize destination under MMRDR-OCT/
            if member.startswith("MMRDR/MMRDR-OCT/"):
                rel = member[len("MMRDR/MMRDR-OCT/") :]
            elif member.startswith("MMRDR-OCT/"):
                rel = member[len("MMRDR-OCT/") :]
            else:
                continue
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 1024)

    if not csv_path.exists():
        raise RuntimeError(f"Extraction finished but missing {csv_path}")
    n_img = len(list((target / "img").glob("*.jpg")))
    print(f"Done: {n_img} images + OCT.csv at {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MMRDR-OCT from Figshare bundle")
    parser.add_argument(
        "--bundle",
        type=Path,
        default=FIGSHARE_BUNDLE,
        help="Path to 29423747.zip (Figshare download)",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Keep merged zip / parts under data/raw/mmrdr/_staging",
    )
    parser.add_argument(
        "--parts-dir",
        type=Path,
        default=None,
        help="If parts already extracted to a folder, use them instead of re-opening the bundle",
    )
    args = parser.parse_args()

    MMRDR_DIR.mkdir(parents=True, exist_ok=True)
    STAGING.mkdir(parents=True, exist_ok=True)

    if args.parts_dir and args.parts_dir.exists():
        parts = sorted(args.parts_dir.glob("MMRDR.zip.*"))
    elif STAGING.exists() and list(STAGING.glob("MMRDR.zip.*")):
        parts = sorted(STAGING.glob("MMRDR.zip.*"))
        print(f"Using existing parts in {STAGING}")
    else:
        if not args.bundle.exists():
            raise SystemExit(f"Bundle not found: {args.bundle}")
        parts = extract_parts_from_bundle(args.bundle, STAGING)

    if not parts:
        raise SystemExit("No MMRDR.zip.* parts available")

    merged = merge_parts(parts, MERGED_ZIP)
    extract_oct_only(merged, MMRDR_DIR)

    if not args.keep_staging:
        # Keep merged zip by default for resume; delete only loose parts to save space
        for part in parts:
            try:
                part.unlink()
            except OSError:
                pass
        print("Removed part files (kept merged zip in _staging). Use --keep-staging to keep parts.")

    print(f"MMRDR ready at {MMRDR_DIR / 'MMRDR-OCT'}")


if __name__ == "__main__":
    main()
