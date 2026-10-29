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
WEIGHTS = {
    "oct": {
        "repo": "monish563/RETFOUND",
        "out": ROOT / "checkpoints" / "weights" / "RETFound_oct.pth",
        "filename": "RETFound_oct_weights.pth",
        "accept_url": "https://huggingface.co/monish563/RETFOUND",
    },
    "cfp": {
        "repo": "YukunZhou/RETFound_mae_natureCFP",
        "out": ROOT / "checkpoints" / "weights" / "RETFound_cfp.pth",
        "filename": "RETFound_mae_natureCFP.pth",
        "accept_url": "https://huggingface.co/YukunZhou/RETFound_mae_natureCFP",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download RETFound OCT weights")
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF read token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--modality",
        choices=["oct", "cfp", "both"],
        default="oct",
        help="Which RETFound weights to download (cfp = colour fundus)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (overrides --modality default)",
    )
    args = parser.parse_args()

    if not args.token:
        print(
            "No HF token found.\n\n"
            "1. Accept access on the HuggingFace model page for each weight\n"
            "   OCT: https://huggingface.co/monish563/RETFOUND\n"
            "   CFP: https://huggingface.co/YukunZhou/RETFound_mae_natureCFP\n"
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

    modalities = ["oct", "cfp"] if args.modality == "both" else [args.modality]
    for mod in modalities:
        spec = WEIGHTS[mod]
        out = args.out if args.out and len(modalities) == 1 else spec["out"]
        filename = spec["filename"]
        repo_id = spec["repo"]
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {repo_id}/{filename} -> {out}")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(out.parent),
            token=args.token,
        )
        downloaded = Path(path)
        if downloaded.resolve() != out.resolve() and downloaded.exists():
            downloaded.replace(out)
        print(f"Saved: {out}")

    print(
        "\nOCT train:  python draco/train.py --config configs/retfound_mmrdr_v2.yaml\n"
        "Fundus train: python glauco/train.py --config configs/glaucoma_retfound_cfp_airogs_light.yaml"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
