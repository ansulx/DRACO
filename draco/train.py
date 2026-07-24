#!/usr/bin/env python3
"""Train a DRACO OCT classifier from a YAML config."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from draco.data.datasets import make_loader
from draco.losses import build_criterion
from draco.metrics import multiclass_metrics
from draco.models.factory import build_model
from draco.optim import adjust_learning_rate, param_groups_lrd


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_lr(cfg: dict, batch_size: int) -> float:
    """Absolute LR, or RETFound-style base LR: lr = blr * batch / 256."""
    if cfg.get("lr") is not None:
        return float(cfg["lr"])
    if cfg.get("blr") is not None:
        return float(cfg["blr"]) * batch_size / 256.0
    return 1e-4


def _build_optimizer(model, cfg: dict, lr: float) -> torch.optim.Optimizer:
    weight_decay = float(cfg.get("weight_decay", 0.05))
    layer_decay = cfg.get("layer_decay")
    if layer_decay is not None and hasattr(model, "blocks"):
        groups = param_groups_lrd(
            model,
            weight_decay=weight_decay,
            no_weight_decay_list=("pos_embed", "cls_token"),
            layer_decay=float(layer_decay),
        )
        optimizer = torch.optim.AdamW(groups, lr=lr)
        print(
            f"Optimizer: AdamW + layer_decay={layer_decay} "
            f"({len(groups)} param groups, peak_lr={lr:.2e})"
        )
        return optimizer
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def _set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze / unfreeze everything except the classification head."""
    for name, param in model.named_parameters():
        if name.startswith("head") or ".head." in name:
            param.requires_grad = True
        else:
            param.requires_grad = trainable


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    ys, preds = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        pred = logits.argmax(dim=1).cpu().numpy()
        ys.append(labels.numpy())
        preds.append(pred)
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    return multiclass_metrics(y_true, y_pred)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp: bool):
    model.train()
    total_loss = 0.0
    n = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
    return total_loss / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="DRACO train")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device(
        args.device
        or cfg.get("device")
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    data_cfg = cfg["data"]
    train_csv = ROOT / data_cfg["train_csv"]
    test_csv = ROOT / data_cfg["test_csv"]
    image_size = int(data_cfg.get("image_size", 224))
    batch_size = int(cfg.get("batch_size", 32))
    num_workers = int(cfg.get("num_workers", 2))
    num_classes = int(cfg.get("num_classes", 3))
    binary = bool(cfg.get("binary", False))

    class_weights = None
    weights_path = ROOT / data_cfg.get("class_weights", "data/processed/stats/class_weights.json")
    if weights_path.exists() and not binary:
        with open(weights_path, encoding="utf-8") as f:
            class_weights = json.load(f)["weights"]

    train_loader = make_loader(
        train_csv,
        split="train",
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        binary=binary,
        class_weights=class_weights,
    )
    test_loader = make_loader(
        test_csv,
        split="eval",
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        binary=binary,
        shuffle=False,
    )

    model = build_model(
        cfg["model"]["name"],
        num_classes=2 if binary else num_classes,
        pretrained=bool(cfg["model"].get("pretrained", True)),
        checkpoint=cfg["model"].get("checkpoint"),
    ).to(device)

    # Fresh classifier head (official RETFound reinits head after loading MAE)
    if hasattr(model, "head") and hasattr(model.head, "weight"):
        nn.init.trunc_normal_(model.head.weight, std=2e-5)
        if model.head.bias is not None:
            nn.init.zeros_(model.head.bias)

    head_warmup = int(cfg.get("head_warmup_epochs", 0))
    if head_warmup > 0:
        _set_backbone_trainable(model, trainable=False)
        print(f"Head-only warmup for first {head_warmup} epochs")

    loss_cfg = cfg.get("loss") or {}
    if isinstance(loss_cfg, str):
        loss_cfg = {"name": loss_cfg}
    loss_name = str(loss_cfg.get("name", "ce"))
    criterion = build_criterion(
        loss_name,
        class_weights=None if binary else class_weights,
        device=device,
        gamma=float(loss_cfg.get("gamma", 2.0)),
        label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
    ).to(device)
    print(
        f"Loss: {loss_name} "
        f"(gamma={loss_cfg.get('gamma', 2.0)}, "
        f"label_smoothing={loss_cfg.get('label_smoothing', 0.0)}, "
        f"class_weights={'yes' if (class_weights and not binary) else 'no'})"
    )

    lr = _resolve_lr(cfg, batch_size)
    optimizer = _build_optimizer(model, cfg, lr)
    epochs = int(cfg.get("epochs", 40))
    patience = int(cfg.get("patience", 10))
    warmup_epochs = int(cfg.get("warmup_epochs", 0))
    min_lr = float(cfg.get("min_lr", 1e-6))
    use_cosine = bool(cfg.get("cosine", warmup_epochs > 0 or cfg.get("blr") is not None))
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(
        f"LR schedule: peak={lr:.2e} warmup={warmup_epochs} "
        f"cosine={use_cosine} min_lr={min_lr:.2e}"
    )

    run_name = cfg.get("run_name", args.config.stem)
    out_dir = ROOT / "checkpoints" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    best_f1 = -1.0
    best_epoch = -1
    history = []
    bad = 0
    backbone_unfrozen = head_warmup <= 0

    for epoch in range(1, epochs + 1):
        if head_warmup > 0 and not backbone_unfrozen and epoch > head_warmup:
            _set_backbone_trainable(model, trainable=True)
            optimizer = _build_optimizer(model, cfg, lr)
            backbone_unfrozen = True
            print(f"Unfroze backbone at epoch {epoch}")

        if use_cosine:
            cur_lr = adjust_learning_rate(
                optimizer,
                epoch - 1,  # RETFound indexes warmup from 0
                lr=lr,
                min_lr=min_lr,
                epochs=epochs,
                warmup_epochs=warmup_epochs,
            )
        else:
            cur_lr = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        metrics = evaluate(model, test_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = train_loss
        metrics["lr"] = cur_lr
        history.append(metrics)
        print(
            f"Epoch {epoch:03d} | loss={train_loss:.4f} | lr={cur_lr:.2e} | "
            f"acc={metrics['accuracy']:.4f} | macro_f1={metrics['macro_f1']:.4f}"
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch
            bad = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                    "config": cfg,
                },
                out_dir / "best.pt",
            )
            with open(out_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Best macro_f1={best_f1:.4f} @ epoch {best_epoch}")
    print(f"Checkpoint: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
