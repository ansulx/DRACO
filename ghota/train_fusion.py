"""Train GHOTA fusion on real OEFI same-eye pairs (specialists frozen)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from draco.metrics import binary_metrics, multiclass_metrics
from ghota.models.factory import build_ghota_from_config
from ghota.models.fusion import CrossAttentionFusion, FeatureFusion, LateFusion
from glauco.data.datasets import load_image
from glauco.data.transforms import build_transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PairedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int, split: str):
        self.df = df.reset_index(drop=True)
        self.tf_o = build_transforms(split, image_size, "oct")
        self.tf_f = build_transforms(split, image_size, "fundus")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        xo = self.tf_o(load_image(Path(row["path_oct"]), "oct"))
        xf = self.tf_f(load_image(Path(row["path_fundus"]), "fundus"))
        y = torch.tensor(int(row["y"]), dtype=torch.long)
        return xo, xf, y


def load_specialist(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config") or yaml.safe_load(
        (ckpt_path.parent / "config.yaml").read_text(encoding="utf-8")
    )
    model = build_ghota_from_config(cfg, load_weights=False).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.freeze()
    return model, cfg


def build_fusion(kind: str, feat_dim: int, device: torch.device) -> nn.Module:
    if kind == "late_learnable":
        return LateFusion(num_classes=3, learnable=True).to(device)
    if kind == "feature":
        return FeatureFusion(dim=feat_dim, num_classes=3).to(device)
    if kind == "cross_attn":
        return CrossAttentionFusion(dim=feat_dim, num_heads=8, num_classes=3).to(device)
    raise ValueError(f"Unknown fusion_type={kind}")


def patient_split(df: pd.DataFrame, test_size: float, seed: int):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups = df["patient"].astype(str).values
    y = df["y"].values
    tr_idx, te_idx = next(gss.split(df, y, groups))
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)


@torch.no_grad()
def _specialist_pack(model_o, model_f, xo, xf, device):
    d0 = torch.zeros(xo.size(0), dtype=torch.long, device=device)
    lo = model_o(xo, d0)
    lf = model_f(xf, d0)
    fo = model_o.forward_features(xo)
    ff = model_f.forward_features(xf)
    to = model_o.forward_tokens(xo)
    tf = model_f.forward_tokens(xf)
    # tokens: keep sequence if 3D; for CNN pool spatial → fake tokens
    if to.ndim == 4:
        b, c, h, w = to.shape
        to = to.flatten(2).transpose(1, 2)  # B,HW,C
    if tf.ndim == 4:
        b, c, h, w = tf.shape
        tf = tf.flatten(2).transpose(1, 2)
    return lo, lf, fo, ff, to, tf


def fusion_forward(fusion, kind, lo, lf, fo, ff, to, tf):
    if kind == "late_learnable":
        return fusion(lo, lf, return_probs=False)
    if kind == "feature":
        return fusion(fo, ff)
    if kind == "cross_attn":
        return fusion(to, tf)
    raise ValueError(kind)


@torch.no_grad()
def evaluate(fusion, kind, model_o, model_f, loader, device):
    fusion.eval()
    ys, preds, probs = [], [], []
    for xo, xf, y in loader:
        xo = xo.to(device, non_blocking=True)
        xf = xf.to(device, non_blocking=True)
        lo, lf, fo, ff, to, tf = _specialist_pack(model_o, model_f, xo, xf, device)
        logits = fusion_forward(fusion, kind, lo, lf, fo, ff, to, tf)
        p = torch.softmax(logits, dim=1)
        ys.append(y.numpy())
        preds.append(p.argmax(1).cpu().numpy())
        probs.append(p.cpu().numpy())
    y = np.concatenate(ys)
    pred = np.concatenate(preds)
    prob = np.concatenate(probs)
    out = multiclass_metrics(y, pred)
    out["binary_dme_collapse"] = binary_metrics(
        (y > 0).astype(int), 1.0 - prob[:, 0]
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="GHOTA paired fusion train")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pairs = pd.read_csv(ROOT / cfg["pairs_csv"])
    target = cfg.get("target", "dr")
    if target == "dr":
        pairs = pairs.dropna(subset=["label_dr_ghota"]).copy()
        pairs["y"] = pairs["label_dr_ghota"].astype(int)
    else:
        pairs["y"] = pairs["label_dme"].astype(int)
        # binary → map to {0,1} but head is 3-class; use 2-class via CE on collapsed
        # Keep as 0/1 labels with 3-class head is messy — force binary CE on 2 logits via FeatureFusion override
        raise SystemExit("Use target=dr for v1 fusion train (DME reported via collapse).")

    print(
        f"Pairs with labels: {len(pairs)} | class counts={pairs['y'].value_counts().sort_index().to_dict()}"
    )
    train_df, test_df = patient_split(pairs, float(cfg.get("test_size", 0.25)), int(cfg["seed"]))
    print(
        f"Patient split: train={len(train_df)} ({train_df.patient.nunique()} pts) "
        f"test={len(test_df)} ({test_df.patient.nunique()} pts)"
    )

    bs = int(cfg.get("batch_size", 4))
    nw = int(cfg.get("num_workers", 2))
    isize = int(cfg.get("image_size", 224))
    train_loader = DataLoader(
        PairedDataset(train_df, isize, "train"),
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        PairedDataset(test_df, isize, "eval"),
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
    )

    model_o, _ = load_specialist(ROOT / cfg["oct_checkpoint"], device)
    model_f, _ = load_specialist(ROOT / cfg["fundus_checkpoint"], device)
    feat_dim = int(model_o.feat_dim)
    kind = str(cfg["fusion_type"])
    fusion = build_fusion(kind, feat_dim, device)
    print(f"Fusion={kind} feat_dim={feat_dim} params={sum(p.numel() for p in fusion.parameters())}")

    opt = torch.optim.AdamW(
        fusion.parameters(),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )
    criterion = nn.CrossEntropyLoss()
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out = ROOT / "checkpoints" / cfg.get("run_name", args.config.stem)
    out.mkdir(parents=True, exist_ok=True)
    # persist split for reproducibility
    train_df.to_csv(out / "train_pairs.csv", index=False)
    test_df.to_csv(out / "test_pairs.csv", index=False)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    # baselines on test (frozen specialists + avg late)
    avg = LateFusion(learnable=False).to(device)
    base_ys, base_po, base_pf, base_fuse = [], [], [], []
    with torch.no_grad():
        for xo, xf, y in test_loader:
            xo, xf = xo.to(device), xf.to(device)
            lo, lf, *_ = _specialist_pack(model_o, model_f, xo, xf, device)
            po = torch.softmax(lo, 1)
            pf = torch.softmax(lf, 1)
            pfuse = avg(lo, lf)
            base_ys.append(y.numpy())
            base_po.append(po.argmax(1).cpu().numpy())
            base_pf.append(pf.argmax(1).cpu().numpy())
            base_fuse.append(pfuse.argmax(1).cpu().numpy())
    yb = np.concatenate(base_ys)
    baselines = {
        "oct_only": multiclass_metrics(yb, np.concatenate(base_po)),
        "fundus_only": multiclass_metrics(yb, np.concatenate(base_pf)),
        "avg_late_fusion": multiclass_metrics(yb, np.concatenate(base_fuse)),
    }
    print(
        "Baselines test: "
        f"OCT={baselines['oct_only']['accuracy']:.3f} "
        f"Fundus={baselines['fundus_only']['accuracy']:.3f} "
        f"AvgFuse={baselines['avg_late_fusion']['accuracy']:.3f}"
    )

    best, bad, history = -1.0, 0, []
    epochs = int(cfg.get("epochs", 40))
    patience = int(cfg.get("patience", 12))

    for epoch in range(1, epochs + 1):
        fusion.train()
        total, n = 0.0, 0
        for xo, xf, y in train_loader:
            xo = xo.to(device, non_blocking=True)
            xf = xf.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.no_grad():
                lo, lf, fo, ff, to, tf = _specialist_pack(model_o, model_f, xo, xf, device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = fusion_forward(fusion, kind, lo, lf, fo, ff, to, tf)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * y.size(0)
            n += y.size(0)

        metrics = evaluate(fusion, kind, model_o, model_f, test_loader, device)
        metrics.update(epoch=epoch, train_loss=total / max(n, 1))
        history.append(metrics)
        print(
            f"Epoch {epoch:03d} | loss={metrics['train_loss']:.4f} | "
            f"acc={metrics['accuracy']:.4f} | macro_f1={metrics['macro_f1']:.4f}"
        )
        score = float(metrics["macro_f1"])
        if score > best:
            best, bad = score, 0
            payload = {
                "fusion": fusion.state_dict(),
                "fusion_type": kind,
                "epoch": epoch,
                "metrics": metrics,
                "baselines": baselines,
                "config": cfg,
                "feat_dim": feat_dim,
            }
            torch.save(payload, out / "best.pt")
            (out / "best_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {epoch} (best F1={best:.4f})")
                break

    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out / "baselines.json").write_text(json.dumps(baselines, indent=2), encoding="utf-8")
    print(f"Best macro_f1={best:.4f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
