"""GHOTA inference CLI — modality routing + optional paired fusion.

Examples:
  python -m ghota.infer --image scan.jpg --modality oct
  python -m ghota.infer --image photo.jpg --modality fundus
  python -m ghota.infer --oct scan.jpg --fundus photo.jpg
  python -m ghota.infer --oct scan.jpg --fundus photo.jpg --fusion-ckpt checkpoints/ghota_oefi_fusion_feature/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ghota.labels import GHOTA_CLASS_NAMES
from ghota.models.factory import build_ghota_from_config
from ghota.models.fusion import CrossAttentionFusion, FeatureFusion, LateFusion
from glauco.data.datasets import load_image
from glauco.data.transforms import build_transforms

DEFAULT_OCT = ROOT / "checkpoints/ghota_oct_dme_retfound_sota/best.pt"
DEFAULT_FUNDUS = ROOT / "checkpoints/ghota_fundus_dr_retfound_sota/best.pt"
DEFAULT_FUSION = ROOT / "checkpoints/ghota_oefi_fusion_feature/best.pt"


def _load_specialist(ckpt: Path, device: torch.device):
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = blob.get("config") or yaml.safe_load((ckpt.parent / "config.yaml").read_text(encoding="utf-8"))
    model = build_ghota_from_config(cfg, load_weights=False).to(device)
    model.load_state_dict(blob["model"], strict=False)
    model.eval()
    return model, cfg


def _load_fusion(ckpt: Path, feat_dim: int, device: torch.device):
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    kind = blob.get("fusion_type") or blob.get("config", {}).get("fusion_type", "feature")
    if kind == "late_learnable":
        fusion = LateFusion(num_classes=3, learnable=True)
    elif kind == "feature":
        fusion = FeatureFusion(dim=int(blob.get("feat_dim", feat_dim)), num_classes=3)
    elif kind == "cross_attn":
        fusion = CrossAttentionFusion(dim=int(blob.get("feat_dim", feat_dim)), num_classes=3)
    else:
        raise ValueError(f"Unknown fusion_type={kind}")
    fusion.load_state_dict(blob["fusion"])
    fusion.to(device).eval()
    return fusion, kind


def _prep(path: Path, modality: str, image_size: int, device: torch.device):
    img = load_image(path, modality)
    x = build_transforms("eval", image_size, modality)(img).unsqueeze(0).to(device)
    return x


@torch.no_grad()
def predict_single(model, x, device) -> dict:
    d = torch.zeros(1, dtype=torch.long, device=device)
    logits = model(x, d)
    probs = F.softmax(logits, dim=1)[0].cpu().tolist()
    pred = int(logits.argmax(1).item())
    return {
        "pred_class": pred,
        "pred_name": GHOTA_CLASS_NAMES[pred],
        "probs": {GHOTA_CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
        "dme_score": float(1.0 - probs[0]),
    }


@torch.no_grad()
def predict_fused(model_o, model_f, fusion, kind, xo, xf, device) -> dict:
    d = torch.zeros(1, dtype=torch.long, device=device)
    lo = model_o(xo, d)
    lf = model_f(xf, d)
    if kind == "late_learnable":
        logits = fusion(lo, lf, return_probs=False)
    elif kind == "feature":
        logits = fusion(model_o.forward_features(xo), model_f.forward_features(xf))
    else:
        to, tf_ = model_o.forward_tokens(xo), model_f.forward_tokens(xf)
        if to.ndim == 4:
            to = to.flatten(2).transpose(1, 2)
        if tf_.ndim == 4:
            tf_ = tf_.flatten(2).transpose(1, 2)
        logits = fusion(to, tf_)
    probs = F.softmax(logits, dim=1)[0].cpu().tolist()
    pred = int(logits.argmax(1).item())
    out = {
        "pred_class": pred,
        "pred_name": GHOTA_CLASS_NAMES[pred],
        "probs": {GHOTA_CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
        "dme_score": float(1.0 - probs[0]),
        "fusion_type": kind,
        "oct_branch": predict_single(model_o, xo, device),
        "fundus_branch": predict_single(model_f, xf, device),
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="GHOTA infer (route ± fuse)")
    parser.add_argument("--image", type=Path, help="Single image path")
    parser.add_argument("--modality", choices=["oct", "fundus"], help="Required with --image")
    parser.add_argument("--oct", type=Path, help="OCT path for paired fusion")
    parser.add_argument("--fundus", type=Path, help="Fundus path for paired fusion")
    parser.add_argument("--oct-ckpt", type=Path, default=DEFAULT_OCT)
    parser.add_argument("--fundus-ckpt", type=Path, default=DEFAULT_FUNDUS)
    parser.add_argument("--fusion-ckpt", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default=None)
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    paired = args.oct is not None or args.fundus is not None
    single = args.image is not None

    if paired and single:
        print("Use either --image/--modality OR --oct/--fundus, not both.", file=sys.stderr)
        return 2
    if single:
        if not args.modality:
            print("--modality required with --image", file=sys.stderr)
            return 2
        if not args.image.exists():
            print(f"Missing image: {args.image}", file=sys.stderr)
            return 2
        ckpt = args.oct_ckpt if args.modality == "oct" else args.fundus_ckpt
        model, _ = _load_specialist(ckpt, device)
        x = _prep(args.image, args.modality, args.image_size, device)
        result = {
            "mode": "specialist",
            "modality": args.modality,
            "checkpoint": str(ckpt),
            **predict_single(model, x, device),
        }
    elif paired:
        if args.oct is None or args.fundus is None:
            print("Paired mode needs both --oct and --fundus", file=sys.stderr)
            return 2
        if not args.oct.exists() or not args.fundus.exists():
            print("OCT/fundus path missing", file=sys.stderr)
            return 2
        model_o, _ = _load_specialist(args.oct_ckpt, device)
        model_f, _ = _load_specialist(args.fundus_ckpt, device)
        xo = _prep(args.oct, "oct", args.image_size, device)
        xf = _prep(args.fundus, "fundus", args.image_size, device)
        fusion_ckpt = args.fusion_ckpt or (DEFAULT_FUSION if DEFAULT_FUSION.exists() else None)
        if fusion_ckpt and fusion_ckpt.exists():
            fusion, kind = _load_fusion(fusion_ckpt, model_o.feat_dim, device)
            result = {
                "mode": "paired_fusion",
                "fusion_checkpoint": str(fusion_ckpt),
                **predict_fused(model_o, model_f, fusion, kind, xo, xf, device),
            }
        else:
            # honest fallback: average late fusion
            fusion = LateFusion(learnable=False).to(device)
            lo = model_o(xo, torch.zeros(1, dtype=torch.long, device=device))
            lf = model_f(xf, torch.zeros(1, dtype=torch.long, device=device))
            probs = fusion(lo, lf)[0].cpu().tolist()
            pred = int(max(range(3), key=lambda i: probs[i]))
            result = {
                "mode": "paired_avg_late",
                "note": "No trained fusion ckpt found; using 0.5 logit average",
                "pred_class": pred,
                "pred_name": GHOTA_CLASS_NAMES[pred],
                "probs": {GHOTA_CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
                "dme_score": float(1.0 - probs[0]),
                "oct_branch": predict_single(model_o, xo, device),
                "fundus_branch": predict_single(model_f, xf, device),
            }
    else:
        parser.print_help()
        return 2

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"mode: {result['mode']}")
        print(f"pred: {result['pred_class']} ({result['pred_name']})")
        print(f"dme_score: {result['dme_score']:.4f}")
        print("probs:", ", ".join(f"{k}={v:.3f}" for k, v in result["probs"].items()))
        if "oct_branch" in result:
            print(
                f"  oct_branch: {result['oct_branch']['pred_name']} "
                f"({result['oct_branch']['probs']})"
            )
            print(
                f"  fundus_branch: {result['fundus_branch']['pred_name']} "
                f"({result['fundus_branch']['probs']})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
