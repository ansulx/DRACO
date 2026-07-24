#!/usr/bin/env python3
"""Evaluate a trained DRACO checkpoint on MMRDR and/or OEFI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from draco.data.datasets import make_loader
from draco.metrics import binary_metrics, multiclass_metrics, remap_3class_to_binary_prob
from draco.models.factory import build_model


@torch.no_grad()
def collect_outputs(model, loader, device):
    model.eval()
    ys, preds, probs = [], [], []
    for images, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        p = torch.softmax(logits, dim=1).cpu().numpy()
        pred = p.argmax(axis=1)
        ys.append(labels.numpy())
        preds.append(pred)
        probs.append(p)
    return np.concatenate(ys), np.concatenate(preds), np.concatenate(probs)


def load_checkpoint(path: Path, device):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("config") or {}
    if not cfg and (path.parent / "config.yaml").exists():
        with open(path.parent / "config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    return ckpt, cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="DRACO evaluate")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mmrdr-test", type=Path, default=ROOT / "data/processed/catalogs/mmrdr_test.csv")
    parser.add_argument("--oefi", type=Path, default=ROOT / "data/processed/catalogs/oefi_external.csv")
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
    model_name = cfg.get("model", {}).get("name", "efficientnet_b0")
    num_classes = int(cfg.get("num_classes", 3))
    binary = bool(cfg.get("binary", False))

    model = build_model(
        model_name,
        num_classes=2 if binary else num_classes,
        pretrained=False,
        checkpoint=None,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)

    report = {"checkpoint": str(args.checkpoint), "device": str(device)}

    if args.mmrdr_test.exists():
        loader = make_loader(
            args.mmrdr_test,
            split="eval",
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers,
            binary=binary,
            shuffle=False,
        )
        y_true, y_pred, probs = collect_outputs(model, loader, device)
        if binary:
            # OEFI-style binary metrics on MMRDR remapped
            y_prob = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
            report["mmrdr"] = binary_metrics(y_true, y_prob)
        else:
            report["mmrdr"] = multiclass_metrics(y_true, y_pred)
            # Also report remapped binary AUC for comparison
            y_bin = (y_true > 0).astype(int)
            report["mmrdr_binary_remapped"] = binary_metrics(
                y_bin, remap_3class_to_binary_prob(probs)
            )
        print("MMRDR:", json.dumps(report["mmrdr"], indent=2))

    if args.oefi.exists():
        loader = make_loader(
            args.oefi,
            split="eval",
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers,
            binary=False,  # OEFI labels already 0/1
            shuffle=False,
        )
        y_true, y_pred, probs = collect_outputs(model, loader, device)
        if binary or probs.shape[1] == 2:
            y_prob = probs[:, 1]
        else:
            # 3-class model -> P(DME)=P1+P2
            y_prob = remap_3class_to_binary_prob(probs)
        report["oefi"] = binary_metrics(y_true, y_prob)
        print("OEFI:", json.dumps(report["oefi"], indent=2))

    out = args.out or (args.checkpoint.parent / "eval_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
