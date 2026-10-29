"""Unpaired / paired DR fusion for GHOTA.

Production rule: fuse only on real same-eye pairs. Specialists stay frozen;
fusion modules are trained on OEFI-style paired catalogs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LateFusion(nn.Module):
    """Gate over specialist logits (fixed 0.5 or learnable α)."""

    def __init__(self, num_classes: int = 3, learnable: bool = False):
        super().__init__()
        self.learnable = learnable
        self.num_classes = num_classes
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(0.0))  # sigmoid→0.5 init

    def forward(
        self,
        logits_oct: torch.Tensor,
        logits_fundus: torch.Tensor,
        *,
        return_probs: bool = True,
    ) -> torch.Tensor:
        if self.learnable:
            a = torch.sigmoid(self.alpha)
            logits = a * logits_oct + (1.0 - a) * logits_fundus
        else:
            logits = 0.5 * (logits_oct + logits_fundus)
        return F.softmax(logits, dim=-1) if return_probs else logits


class FeatureFusion(nn.Module):
    """MLP on concatenated CLS features from frozen OCT + fundus specialists."""

    def __init__(self, dim: int, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        hidden = max(dim // 2, 128)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, feat_oct: torch.Tensor, feat_fundus: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([feat_oct, feat_fundus], dim=-1))


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention over OCT/fundus token sequences (B,T,D)."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1, num_classes: int = 3):
        super().__init__()
        self.oct_to_fundus = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.fundus_to_oct = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_o = nn.LayerNorm(dim)
        self.norm_f = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, num_classes),
        )

    def forward(
        self, tokens_oct: torch.Tensor, tokens_fundus: torch.Tensor
    ) -> torch.Tensor:
        # If CLS-only (B,D), promote to (B,1,D)
        if tokens_oct.ndim == 2:
            tokens_oct = tokens_oct.unsqueeze(1)
        if tokens_fundus.ndim == 2:
            tokens_fundus = tokens_fundus.unsqueeze(1)
        o2f, _ = self.oct_to_fundus(tokens_oct, tokens_fundus, tokens_fundus)
        f2o, _ = self.fundus_to_oct(tokens_fundus, tokens_oct, tokens_oct)
        o = self.norm_o(tokens_oct + o2f)[:, 0]
        f = self.norm_f(tokens_fundus + f2o)[:, 0]
        return self.head(torch.cat([o, f], dim=-1))
