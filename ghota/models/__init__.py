"""Disease-conditioned 3-class GHOTA classifier."""

from __future__ import annotations

import torch
import torch.nn as nn

from ghota.labels import DISEASES, NUM_CLASSES


class GhotaModel(nn.Module):
    """Shared vision backbone + disease embedding → 3-class severity.

    ``disease_id`` is an integer index into ``DISEASES`` (or a longer registry).
    """

    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        num_diseases: int = len(DISEASES),
        num_classes: int = NUM_CLASSES,
        disease_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.disease_emb = nn.Embedding(num_diseases, disease_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim + disease_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim + disease_dim, num_classes),
        )
        self.feat_dim = feat_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_tokens(x)
        if tokens.ndim == 4:  # CNN (B,C,H,W)
            return tokens.mean(dim=(2, 3))
        if tokens.ndim == 3:  # ViT tokens (B,T,D)
            return tokens[:, 0] if tokens.size(1) > 1 else tokens.mean(dim=1)
        return tokens

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Raw backbone tokens / feature map (no disease emb)."""
        feats = getattr(self.backbone, "forward_features", None)
        if feats is not None:
            return feats(x)
        return self.backbone(x)

    def forward(self, x: torch.Tensor, disease_id: torch.Tensor) -> torch.Tensor:
        h = self.forward_features(x)
        d = self.disease_emb(disease_id.long())
        return self.head(torch.cat([h, d], dim=1))

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = trainable

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
