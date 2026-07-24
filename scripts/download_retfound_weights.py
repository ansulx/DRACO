#!/usr/bin/env python3
"""Download gated RETFound OCT MAE weights from Hugging Face.

Prerequisites:
  1. Hugging Face account
  2. Accept access on https://huggingface.co/monish563/RETFOUND
  3. Read token from https://huggingface.co/settings/tokens

Usage:
  set HF_TOKEN=hf_...
  python scripts/download_retfound_weights.py

  python scripts/download_retfound_weights.py --token hf_...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "checkpoints" / "weights" / "RETFound_oct.pth"
REPO_ID = "monish563/RETFOUND"
FILENAME = "RETFound_oct_weights.pth"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download RETFound OCT weights")
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF read token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output path (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    if not args.token:
        print(
            "No HF token found.\n\n"
            "1. Accept access: https://huggingface.co/monish563/RETFOUND\n"
            "2. Create a read token: https://huggingface.co/settings/tokens\n"
            "3. Run:\n"
            "     set HF_TOKEN=hf_...\n"
            "     python scripts/download_retfound_weights.py\n",
            file=sys.stderr,
        )
        return 1

    try:
        from huggingface_hub import hf_hub_download, login
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
        return 1

    login(token=args.token, add_to_git_credential=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID}/{FILENAME} ...")
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=str(args.out.parent),
        token=args.token,
    )
    downloaded = Path(path)
    if downloaded.resolve() != args.out.resolve() and downloaded.exists():
        downloaded.replace(args.out)

    print(f"Saved: {args.out}")
    print(
        "\nNext: set checkpoint in configs/retfound_mmrdr.yaml:\n"
        "  checkpoint: checkpoints/weights/RETFound_oct.pth\n"
        "Then train:\n"
        "  python draco/train.py --config configs/retfound_mmrdr.yaml"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
