#!/usr/bin/env python3
"""Run a trained GLAUCO checkpoint on one or more images.

Examples:
  # Dual fusion
  python scripts/glauco_infer.py --checkpoint checkpoints/glauco_fusion_gleam_late/best.pt \\
      --oct path/Thickness.jpg --fundus path/SLO.jpg

  # Tri-modal
  python scripts/glauco_infer.py --checkpoint checkpoints/glauco_fusion_gleam_trimodal_xattn/best.pt \\
      --oct Thickness.jpg --fundus SLO.jpg --vf VF.jpg

  # Single-modality
  python scripts/glauco_infer.py --checkpoint checkpoints/glauco_efficientnet_harvard_gf_full/best.pt \\
      --image sample.npy --modality rnflt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from glauco.data.datasets import load_image
from glauco.data.transforms import build_transforms
from glauco.evaluate import load_checkpoint
from glauco.models import build_from_config, is_fusion_config, is_trimodal_config

STAGE_NAMES = {
    2: ["normal", "glaucoma"],
    3: ["normal", "early", "advanced"],
    4: ["normal", "early", "intermediate", "advanced"],
}


def main() -> int:
    p = argparse.ArgumentParser(description="GLAUCO inference")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image", type=Path, default=None, help="Single-modality image")
    p.add_argument("--oct", type=Path, default=None)
    p.add_argument("--fundus", type=Path, default=None)
    p.add_argument("--vf", type=Path, default=None)
    p.add_argument("--modality", default=None)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ckpt, cfg = load_checkpoint(args.checkpoint, device)
    binary = bool(cfg.get("binary", True))
    num_classes = 2 if binary else int(cfg.get("num_classes", 2))
    model = build_from_config(cfg, num_classes=num_classes, load_weights=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    size = int((cfg.get("data") or {}).get("image_size", args.image_size))

    with torch.no_grad():
        if is_fusion_config(cfg):
            if args.oct is None or args.fundus is None:
                raise SystemExit("Fusion checkpoint needs --oct and --fundus")
            oct_t = build_transforms("eval", size, "oct")(
                load_image(args.oct, "oct")
            ).unsqueeze(0)
            fund_t = build_transforms("eval", size, "fundus")(
                load_image(args.fundus, "fundus")
            ).unsqueeze(0)
            if is_trimodal_config(cfg):
                if args.vf is None:
                    raise SystemExit("Tri-modal checkpoint needs --vf")
                vf_t = build_transforms("eval", size, "vf")(
                    load_image(args.vf, "vf")
                ).unsqueeze(0)
                logits = model(
                    oct_t.to(device), fund_t.to(device), vf_t.to(device)
                )
            else:
                logits = model(oct_t.to(device), fund_t.to(device))
        else:
            if args.image is None:
                raise SystemExit("Single-modality checkpoint needs --image")
            modality = args.modality or cfg.get("modality", "oct")
            img = build_transforms("eval", size, modality)(
                load_image(args.image, modality)
            ).unsqueeze(0)
            logits = model(img.to(device))

        probs = F.softmax(logits.float(), dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())

    names = STAGE_NAMES.get(num_classes, [str(i) for i in range(num_classes)])
    out = {
        "checkpoint": str(args.checkpoint),
        "pred_class": pred,
        "pred_name": names[pred] if pred < len(names) else str(pred),
        "probs": {names[i] if i < len(names) else str(i): float(probs[i]) for i in range(len(probs))},
        "epoch": ckpt.get("epoch"),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
