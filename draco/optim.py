"""Optimizer helpers matching official RETFound / MAE fine-tune."""

from __future__ import annotations

import math
from typing import Iterable

import torch


def param_groups_lrd(
    model: torch.nn.Module,
    weight_decay: float = 0.05,
    no_weight_decay_list: Iterable[str] | None = None,
    layer_decay: float = 0.65,
) -> list[dict]:
    """Layer-wise LR decay for ViT (BEiT / RETFound)."""
    no_weight_decay_list = set(no_weight_decay_list or [])
    if hasattr(model, "blocks"):
        num_layers = len(model.blocks) + 1
    else:
        num_layers = 12

    layer_scales = [layer_decay ** (num_layers - i) for i in range(num_layers + 1)]
    param_groups: dict[str, dict] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim == 1 or name in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = _get_layer_id_for_vit(name, num_layers)
        group_name = f"layer_{layer_id}_{g_decay}"
        if group_name not in param_groups:
            param_groups[group_name] = {
                "lr_scale": layer_scales[layer_id],
                "weight_decay": this_decay,
                "params": [],
            }
        param_groups[group_name]["params"].append(param)

    return list(param_groups.values())


def _get_layer_id_for_vit(name: str, num_layers: int) -> int:
    if name in {"cls_token", "pos_embed"} or name.startswith("patch_embed"):
        return 0
    if name.startswith("blocks"):
        return int(name.split(".")[1]) + 1
    return num_layers


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: float,
    *,
    lr: float,
    min_lr: float,
    epochs: int,
    warmup_epochs: int,
) -> float:
    """Cosine schedule with linear warmup (RETFound / MAE)."""
    if epoch < warmup_epochs:
        new_lr = lr * epoch / max(warmup_epochs, 1)
    else:
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        new_lr = min_lr + (lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

    for group in optimizer.param_groups:
        scale = group.get("lr_scale", 1.0)
        group["lr"] = new_lr * scale
    return new_lr
