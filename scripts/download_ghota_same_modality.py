"""Download same-modality GHOTA add-ons: IDRiD (fundus) + optional Kermany (OCT).

Skips OCTA (different modality). OEFI/APTOS/MMRDR/OCTID assumed already local.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
IDRID_DIR = RAW / "idrid"
KERMANY_DIR = RAW / "kermany"

ZENODO_IDRID = "https://zenodo.org/api/records/17219542"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip exists: {dest.name}")
        return
    headers = {"User-Agent": "ghota-dataset-fetch/1.0"}
    with requests.get(url, stream=True, timeout=120, headers=headers) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f, tqdm(
            total=total or None, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        tmp.replace(dest)


def download_idrid() -> None:
    """Fundus ICDR grades — same modality as GHOTA fundus specialist."""
    print("IDRiD Disease Grading (fundus) from Zenodo…")
    meta = requests.get(ZENODO_IDRID, timeout=60).json()
    files = {f["key"]: f for f in meta["files"]}
    # Prefer disease grading zip only
    key = next(
        (k for k in files if "Disease" in k or "Grading" in k or k.startswith("B.")),
        None,
    )
    if key is None:
        raise RuntimeError(f"No Disease Grading zip in Zenodo record: {list(files)}")
    info = files[key]
    url = info["links"]["self"]
    IDRID_DIR.mkdir(parents=True, exist_ok=True)
    zpath = IDRID_DIR / key.replace(" ", "_")
    _download(url, zpath)
    # extract
    extract_to = IDRID_DIR / "disease_grading"
    if not extract_to.exists() or not any(extract_to.rglob("*.jpg")):
        print(f"  extracting {zpath.name}…")
        extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(extract_to)
    n = len(list(extract_to.rglob("*.jpg")) + list(extract_to.rglob("*.JPG")))
    print(f"IDRiD ready: {n} jpg under {extract_to}")


def download_kermany_kaggle() -> None:
    """OCT B-scans — same modality as GHOTA OCT (large ~5–8GB)."""
    KERMANY_DIR.mkdir(parents=True, exist_ok=True)
    # Prefer credentials.json → kaggle.json if needed (without printing secrets)
    kaggle_dir = Path.home() / ".kaggle"
    cred = kaggle_dir / "credentials.json"
    kj = kaggle_dir / "kaggle.json"
    if cred.exists() and not kj.exists():
        shutil.copy2(cred, kj)
        try:
            kj.chmod(0o600)
        except Exception:
            pass
    print("Kermany OCT via Kaggle paultimothymooney/kermany2018…")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "kaggle",
                "datasets",
                "download",
                "-d",
                "paultimothymooney/kermany2018",
                "-p",
                str(KERMANY_DIR),
                "--unzip",
            ],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(
            "Kaggle download failed. Accept the dataset license in browser, ensure "
            f"kaggle.json exists, then re-run.\n  Error: {e}\n"
            "  Manual: https://www.kaggle.com/datasets/paultimothymooney/kermany2018"
        )
        return
    n = len(list(KERMANY_DIR.rglob("*.jpeg")) + list(KERMANY_DIR.rglob("*.jpg")))
    print(f"Kermany ready: {n} images under {KERMANY_DIR}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        choices=["idrid", "kermany", "all"],
        default="all",
    )
    args = p.parse_args()
    if args.dataset in ("idrid", "all"):
        download_idrid()
    if args.dataset in ("kermany", "all"):
        download_kermany_kaggle()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
