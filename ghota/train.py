"""Train GHOTA DR 3-class with SOTA RETFound recipe (focal + layer-decay + cosine)."""

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
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from draco.losses import build_criterion
from draco.metrics import multiclass_metrics
from draco.optim import adjust_learning_rate, param_groups_lrd
from ghota.data.datasets import GhotaCatalogDataset
from ghota.models.factory import build_ghota_from_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_lr(cfg: dict, batch_size: int) -> float:
    if cfg.get("lr") is not None:
        return float(cfg["lr"])
    if cfg.get("blr") is not None:
        return float(cfg["blr"]) * batch_size / 256.0
    return 1e-4


def _build_optimizer(model: nn.Module, cfg: dict, lr: float) -> torch.optim.Optimizer:
    weight_decay = float(cfg.get("weight_decay", 0.05))
    layer_decay = cfg.get("layer_decay")
    backbone = getattr(model, "backbone", None)
    if layer_decay is not None and backbone is not None and hasattr(backbone, "blocks"):
        # Layer-wise LR on ViT backbone (RETFound / MAE); task emb + head at full lr
        groups = param_groups_lrd(
            backbone,
            weight_decay=weight_decay,
            no_weight_decay_list=("pos_embed", "cls_token"),
            layer_decay=float(layer_decay),
        )
        head_params = [p for p in model.disease_emb.parameters() if p.requires_grad]
        head_params += [p for p in model.head.parameters() if p.requires_grad]
        if head_params:
            groups.append(
                {"params": head_params, "weight_decay": weight_decay, "lr_scale": 1.0}
            )
        print(
            f"Optimizer: AdamW + layer_decay={layer_decay} "
            f"({len(groups)} groups, peak={lr:.2e})"
        )
        return torch.optim.AdamW(groups, lr=lr)
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )


def _warm_start_backbone(model: nn.Module, path: Path) -> None:
    """Load a DRACO/RETFound fine-tune into GhotaModel.backbone (skip classifier)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    backbone_sd = {
        k: v
        for k, v in sd.items()
        if not k.startswith("head") and not k.startswith("disease_emb")
    }
    # DRACO checkpoints are flat ViT keys; GHOTA keeps them under .backbone
    missing, unexpected = model.backbone.load_state_dict(backbone_sd, strict=False)
    print(
        f"Warm-start backbone from {path.name}: "
        f"loaded={len(backbone_sd)} missing={len(missing)} unexpected={len(unexpected)}"
    )


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    ys, preds = [], []
    for x, d, y in loader:
        logits = model(x.to(device, non_blocking=True), d.to(device, non_blocking=True))
        ys.append(y.numpy())
        preds.append(logits.argmax(1).cpu().numpy())
    return multiclass_metrics(np.concatenate(ys), np.concatenate(preds))


def main() -> None:
    parser = argparse.ArgumentParser(description="GHOTA SOTA train")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(
        cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    data_cfg = cfg["data"]
    diseases = list(cfg.get("diseases") or [])
    train_ds = GhotaCatalogDataset(
        ROOT / data_cfg["train_csv"],
        diseases=diseases or None,
        image_size=int(data_cfg.get("image_size", 224)),
        split="train",
        remap=bool(data_cfg.get("remap", True)),
    )
    test_ds = GhotaCatalogDataset(
        ROOT / data_cfg["test_csv"],
        diseases=train_ds.diseases,
        image_size=int(data_cfg.get("image_size", 224)),
        split="eval",
        remap=bool(data_cfg.get("remap", True)),
    )
    cfg["diseases"] = train_ds.diseases
    print(f"Tasks: {train_ds.diseases} | train={len(train_ds)} test={len(test_ds)}")

    class_weights = None
    wp = data_cfg.get("class_weights")
    if wp:
        wpath = ROOT / wp
        if wpath.exists():
            class_weights = json.loads(wpath.read_text(encoding="utf-8")).get("weights")

    sampler, shuffle = None, True
    if cfg.get("balance_classes"):
        counts = np.bincount(train_ds.labels, minlength=3).astype(float)
        w = 1.0 / np.maximum(counts, 1.0)
        sample_w = [float(w[y]) for y in train_ds.labels]
        sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)
        shuffle = False
        print(f"Class counts: {counts.tolist()} | balanced sampler on")

    bs = int(cfg.get("batch_size", 8))
    nw = int(cfg.get("num_workers", 2))
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=shuffle, sampler=sampler,
        num_workers=nw, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True,
    )

    model = build_ghota_from_config(cfg).to(device)
    init_from = cfg.get("init_from") or (cfg.get("model") or {}).get("init_from")
    if init_from:
        _warm_start_backbone(model, ROOT / init_from)
    head_warmup = int(cfg.get("head_warmup_epochs", 0))
    if head_warmup > 0:
        model.set_backbone_trainable(False)
        print(f"Head-only warmup for {head_warmup} epochs")

    loss_cfg = cfg.get("loss") or {"name": "focal"}
    if isinstance(loss_cfg, str):
        loss_cfg = {"name": loss_cfg}
    criterion = build_criterion(
        str(loss_cfg.get("name", "focal")),
        class_weights=class_weights,
        device=device,
        gamma=float(loss_cfg.get("gamma", 2.0)),
        label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
    ).to(device)
    print(f"Loss: {loss_cfg.get('name')} gamma={loss_cfg.get('gamma', 2.0)}")

    lr = _resolve_lr(cfg, bs)
    opt = _build_optimizer(model, cfg, lr)
    epochs = int(cfg.get("epochs", 50))
    patience = int(cfg.get("patience", 15))
    warmup_epochs = int(cfg.get("warmup_epochs", 0))
    min_lr = float(cfg.get("min_lr", 1e-6))
    use_cosine = bool(cfg.get("cosine", True))
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    monitor = str(cfg.get("monitor", "macro_f1"))

    out = ROOT / "checkpoints" / cfg.get("run_name", args.config.stem)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    best, bad, history = -1.0, 0, []
    backbone_unfrozen = head_warmup <= 0
    print(f"LR peak={lr:.2e} cosine={use_cosine} warmup={warmup_epochs} monitor={monitor}")

    for epoch in range(1, epochs + 1):
        if head_warmup > 0 and not backbone_unfrozen and epoch > head_warmup:
            model.set_backbone_trainable(True)
            opt = _build_optimizer(model, cfg, lr)
            backbone_unfrozen = True
            print(f"Unfroze backbone at epoch {epoch}")

        if use_cosine:
            cur_lr = adjust_learning_rate(
                opt, epoch - 1, lr=lr, min_lr=min_lr,
                epochs=epochs, warmup_epochs=warmup_epochs,
            )
        else:
            cur_lr = opt.param_groups[0]["lr"]

        model.train()
        total, n = 0.0, 0
        for x, d, y in train_loader:
            x = x.to(device, non_blocking=True)
            d = d.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(x, d), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * y.size(0)
            n += y.size(0)

        metrics = evaluate(model, test_loader, device)
        metrics.update(epoch=epoch, train_loss=total / max(n, 1), lr=cur_lr)
        history.append(metrics)
        print(
            f"Epoch {epoch:03d} | loss={metrics['train_loss']:.4f} | lr={cur_lr:.2e} | "
            f"acc={metrics['accuracy']:.4f} | macro_f1={metrics['macro_f1']:.4f}"
        )

        score = float(metrics.get(monitor, metrics["macro_f1"]))
        if score > best:
            best, bad = score, 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "config": cfg, "metrics": metrics},
                out / "best.pt",
            )
            (out / "best_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {epoch} (best {best:.4f})")
                break

    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best {monitor}={best:.4f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
