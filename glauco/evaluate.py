#!/usr/bin/env python3
"""Evaluate a trained GLAUCO checkpoint with screening-oriented binary metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, brier_score_loss, roc_curve
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from draco.metrics import binary_metrics, multiclass_metrics
from glauco.data.datasets import make_loader, make_paired_loader, make_trimodal_loader
from glauco.models import build_from_config, is_fusion_config, is_trimodal_config

TARGET_SPECIFICITIES = (0.85, 0.90, 0.95)


def sens_at_spec(
    y_true: np.ndarray, y_prob: np.ndarray, target_spec: float = 0.95
) -> tuple[float, float]:
    """Highest sensitivity (and its threshold) at specificity >= target."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity = 1.0 - fpr
    ok = specificity >= target_spec
    if not np.any(ok):
        idx = int(np.argmax(specificity))
    else:
        idx = int(np.argmax(np.where(ok, tpr, -1.0)))
    thr = float(thresholds[idx])
    if not np.isfinite(thr):  # sklearn prepends +inf for the all-negative point
        thr = 1.0
    return float(tpr[idx]), thr


def glaucoma_binary_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    """draco binary metrics plus operating points a screening reader needs."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    out = dict(binary_metrics(y_true, y_prob, threshold=threshold))
    out["threshold"] = float(threshold)

    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    out.update(
        {
            "sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "ppv": tp / max(tp + fp, 1),
            "npv": tn / max(tn + fn, 1),
            "confusion_matrix": [[tn, fp], [fn, tp]],
        }
    )

    if len(np.unique(y_true)) > 1:
        out["average_precision"] = float(average_precision_score(y_true, y_prob))
        out["brier"] = float(brier_score_loss(y_true, y_prob))
        for spec in TARGET_SPECIFICITIES:
            sens, thr = sens_at_spec(y_true, y_prob, spec)
            tag = f"{int(round(spec * 100))}"
            out[f"sensitivity_at_{tag}_spec"] = sens
            out[f"threshold_at_{tag}_spec"] = thr
    else:
        out["average_precision"] = None
        out["brier"] = None
    return out


@torch.no_grad()
def collect_outputs(model, loader, device):
    model.eval()
    ys, probs = [], []
    for inputs, labels in tqdm(loader, desc="eval", leave=False):
        if isinstance(inputs, (list, tuple)):
            logits = model(*[x.to(device, non_blocking=True) for x in inputs])
        else:
            logits = model(inputs.to(device, non_blocking=True))
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        ys.append(labels.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def load_checkpoint(path: Path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config") or {}
    if not cfg and (path.parent / "config.yaml").exists():
        with open(path.parent / "config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    return ckpt, cfg


def _make_eval_loader(cfg: dict, csv: Path, args, binary: bool):
    common = dict(
        split="eval",
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        binary=binary,
        root=(cfg.get("data") or {}).get("root"),
        shuffle=False,
    )
    if is_fusion_config(cfg):
        if is_trimodal_config(cfg):
            return make_trimodal_loader(csv, **common)
        return make_paired_loader(csv, **common)
    return make_loader(csv, modality=args.modality or cfg.get("modality", "oct"), **common)


def main() -> None:
    parser = argparse.ArgumentParser(description="GLAUCO evaluate")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--test-csv",
        type=Path,
        action="append",
        default=None,
        help="Catalog(s) to evaluate; repeatable. Defaults to the config test_csv.",
    )
    parser.add_argument("--modality", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ckpt, cfg = load_checkpoint(args.checkpoint, device)
    binary = bool(cfg.get("binary", True))
    num_classes = 2 if binary else int(cfg.get("num_classes", 2))

    model = build_from_config(cfg, num_classes=num_classes, load_weights=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    csvs = args.test_csv or [ROOT / cfg["data"]["test_csv"]]
    report: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "epoch": ckpt.get("epoch"),
        "num_classes": num_classes,
    }

    for csv in csvs:
        csv = Path(csv)
        if not csv.is_absolute():
            csv = ROOT / csv
        if not csv.exists():
            print(f"Skipping missing catalog: {csv}")
            continue

        loader = _make_eval_loader(cfg, csv, args, binary)
        y_true, probs = collect_outputs(model, loader, device)

        if num_classes == 2:
            result = glaucoma_binary_metrics(
                y_true, probs[:, 1], threshold=args.threshold
            )
        else:
            result = multiclass_metrics(y_true, probs.argmax(axis=1))
            # Any-glaucoma screening view of a graded model.
            result["binary_remapped"] = glaucoma_binary_metrics(
                (y_true > 0).astype(int), probs[:, 1:].sum(axis=1), threshold=args.threshold
            )

        report[csv.stem] = result
        print(f"{csv.stem}:", json.dumps(result, indent=2))

    out = args.out or (args.checkpoint.parent / "eval_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
