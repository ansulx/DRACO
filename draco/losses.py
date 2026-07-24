"""Loss functions for DRACO training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional class weights.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma > 0 focuses training on hard / misclassified examples.
    alpha is typically inverse-frequency class weights.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)
        if weight is not None:
            self.register_buffer("weight", weight.float())
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # CE per-sample, then modulate by (1 - p_t)^gamma
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        # p_t = exp(-CE_unweighted_for_prob); use softmax of true class
        log_probs = F.log_softmax(logits, dim=1)
        pt = log_probs.gather(1, targets.unsqueeze(1)).exp().squeeze(1).clamp(min=1e-7)
        focal = (1.0 - pt).pow(self.gamma) * ce

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


def build_criterion(
    name: str,
    *,
    class_weights: list[float] | None = None,
    device: torch.device,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """Build loss from config name: ce | focal."""
    name = (name or "ce").lower()
    weight_t = None
    if class_weights is not None:
        weight_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    if name in {"ce", "cross_entropy", "crossentropy"}:
        return nn.CrossEntropyLoss(weight=weight_t, label_smoothing=label_smoothing)

    if name in {"focal", "focal_loss"}:
        return FocalLoss(
            weight=weight_t,
            gamma=gamma,
            label_smoothing=label_smoothing,
        )

    raise ValueError(f"Unknown loss: {name}. Use 'ce' or 'focal'.")
