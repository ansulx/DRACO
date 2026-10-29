#!/usr/bin/env python3
"""After Harvard-GF finishes downloading: extract zip, build catalogs, train.

Usage:
  python scripts/run_harvard_gf_pipeline.py
  python scripts/run_harvard_gf_pipeline.py --train-only   # skip if catalogs exist
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
GF_ROOT = ROOT / "data" / "raw" / "glaucoma" / "harvard_gf"
ZIP = GF_ROOT / "Dataset" / "dataset.zip"
STATUS = ROOT / "results" / "harvard_gf_pipeline.json"


def write_status(**kwargs) -> None:
    data = {"updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if STATUS.exists():
        with open(STATUS, encoding="utf-8") as f:
            data.update(json.load(f))
    data.update(kwargs)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run(args: list[str]) -> int:
    print("+", " ".join(args))
    return subprocess.run([str(PYTHON), *args], cwd=str(ROOT)).returncode


def wait_for_download(timeout_hours: float = 6.0) -> bool:
    deadline = time.time() + timeout_hours * 3600
    while time.time() < deadline:
        if ZIP.exists() and ZIP.stat().st_size > 1_000_000_000:
            return True
        mb = ZIP.stat().st_size / 1e6 if ZIP.exists() else 0
        write_status(phase="waiting_download", zip_mb=round(mb, 1))
        print(f"Waiting for dataset.zip... {mb:.0f} MB")
        time.sleep(60)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--skip-wait", action="store_true")
    args = parser.parse_args()

    write_status(phase="starting")

    if not args.skip_wait and not args.train_only:
        if not ZIP.exists() or ZIP.stat().st_size < 1_000_000_000:
            print("Harvard-GF zip not ready — waiting for HF download...")
            if not wait_for_download():
                write_status(phase="error", error="download timeout")
                return 1

    write_status(phase="build_catalog")
    if run(["scripts/build_glaucoma_catalogs.py", "--dataset", "harvard_gf"]) != 0:
        write_status(phase="error", error="catalog build failed")
        return 1

    write_status(phase="training")
    if run(["glauco/train.py", "--config", "configs/glaucoma_efficientnet_harvard_gf_full.yaml"]) != 0:
        write_status(phase="error", error="training failed")
        return 1

    write_status(phase="eval_official_test")
    ckpt = ROOT / "checkpoints" / "glauco_efficientnet_harvard_gf_full" / "best.pt"
    run(
        [
            "glauco/evaluate.py",
            "--checkpoint",
            str(ckpt),
            "--test-csv",
            "data/processed/glaucoma/catalogs/harvard_gf_test.csv",
            "--modality",
            "rnflt",
            "--out",
            str(ckpt.parent / "eval_official_test.json"),
        ]
    )

    write_status(phase="complete")
    print("Harvard-GF pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
