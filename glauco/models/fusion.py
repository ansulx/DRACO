"""Cross-modal OCT + fundus (+ optional VF) fusion models for glaucoma."""

from __future__ import annotations

import torch
import torch.nn as nn

from glauco.models.factory import build_backbone


class TokenBackbone(nn.Module):
    """Wrap a timm backbone so it emits ``(B, T, D)`` feature tokens.

    CNN feature maps are flattened over space, ViT features are used as-is, and
    already-pooled vectors become a single token. Frozen backbones are held in
    eval mode so BatchNorm statistics stay fixed.
    """

    def __init__(self, backbone: nn.Module, feat_dim: int, trainable: bool = True):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = int(feat_dim)
        self.trainable = True
        self.set_trainable(trainable)

    def set_trainable(self, trainable: bool) -> None:
        self.trainable = bool(trainable)
        for param in self.backbone.parameters():
            param.requires_grad = self.trainable
        if not self.trainable:
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.trainable:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forward_features = getattr(self.backbone, "forward_features", None)
        feats = forward_features(x) if forward_features else self.backbone(x)
        if feats.ndim == 4:  # CNN: (B, C, H, W) -> (B, H*W, C)
            feats = feats.flatten(2).transpose(1, 2)
        elif feats.ndim == 2:  # pooled: (B, D) -> (B, 1, D)
            feats = feats.unsqueeze(1)
        return feats


class _FusionBase(nn.Module):
    """Shared freezing / optimizer-group plumbing for fusion models."""

    def __init__(self, *backbones: TokenBackbone):
        super().__init__()
        self._backbones = nn.ModuleList(backbones)
        # Keep named aliases used by dual-modal subclasses / checkpoints.
        self.oct_backbone = backbones[0] if len(backbones) > 0 else None
        self.fundus_backbone = backbones[1] if len(backbones) > 1 else None
        self.vf_backbone = backbones[2] if len(backbones) > 2 else None

    def set_backbone_trainable(self, trainable: bool) -> None:
        for bb in self._backbones:
            bb.set_trainable(trainable)

    def param_groups(
        self,
        lr: float,
        weight_decay: float = 0.05,
        backbone_lr_scale: float = 1.0,
    ) -> list[dict]:
        """Lower LR for pretrained encoders, full LR for the fusion stack."""
        backbone_ids = {id(p) for m in self._backbones for p in m.parameters()}
        backbone, fusion = [], []
        for param in self.parameters():
            if not param.requires_grad:
                continue
            (backbone if id(param) in backbone_ids else fusion).append(param)

        groups = []
        if backbone:
            groups.append(
                {
                    "params": backbone,
                    "lr": lr * backbone_lr_scale,
                    "lr_scale": backbone_lr_scale,
                    "weight_decay": weight_decay,
                }
            )
        if fusion:
            groups.append(
                {
                    "params": fusion,
                    "lr": lr,
                    "lr_scale": 1.0,
                    "weight_decay": weight_decay,
                }
            )
        return groups


class CrossAttentionFusionModel(_FusionBase):
    """Bidirectional multi-head cross-attention between OCT and fundus tokens."""

    def __init__(
        self,
        oct_backbone: TokenBackbone,
        fundus_backbone: TokenBackbone,
        num_classes: int = 2,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__(oct_backbone, fundus_backbone)
        self.bidirectional = bool(bidirectional)

        self.oct_proj = _projection(oct_backbone.feat_dim, embed_dim)
        self.fundus_proj = _projection(fundus_backbone.feat_dim, embed_dim)

        self.oct_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.oct_norm = nn.LayerNorm(embed_dim)
        if self.bidirectional:
            self.fundus_attn = nn.MultiheadAttention(
                embed_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.fundus_norm = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Dropout(dropout),
            nn.Linear(2 * embed_dim, num_classes),
        )

    def forward(self, oct_image: torch.Tensor, fundus_image: torch.Tensor) -> torch.Tensor:
        oct_tokens = self.oct_proj(self.oct_backbone(oct_image))
        fundus_tokens = self.fundus_proj(self.fundus_backbone(fundus_image))

        oct_ctx, _ = self.oct_attn(oct_tokens, fundus_tokens, fundus_tokens)
        oct_out = self.oct_norm(oct_tokens + oct_ctx)

        if self.bidirectional:
            fundus_ctx, _ = self.fundus_attn(fundus_tokens, oct_tokens, oct_tokens)
            fundus_out = self.fundus_norm(fundus_tokens + fundus_ctx)
        else:
            fundus_out = fundus_tokens

        pooled = torch.cat([oct_out.mean(dim=1), fundus_out.mean(dim=1)], dim=1)
        return self.head(pooled)


class LateFusionModel(_FusionBase):
    """Concatenate mean-pooled per-modality features — the fusion baseline."""

    def __init__(
        self,
        oct_backbone: TokenBackbone,
        fundus_backbone: TokenBackbone,
        num_classes: int = 2,
        embed_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__(oct_backbone, fundus_backbone)
        self.oct_proj = _projection(oct_backbone.feat_dim, embed_dim)
        self.fundus_proj = _projection(fundus_backbone.feat_dim, embed_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Dropout(dropout),
            nn.Linear(2 * embed_dim, num_classes),
        )

    def forward(self, oct_image: torch.Tensor, fundus_image: torch.Tensor) -> torch.Tensor:
        oct_feat = self.oct_proj(self.oct_backbone(oct_image)).mean(dim=1)
        fundus_feat = self.fundus_proj(self.fundus_backbone(fundus_image)).mean(dim=1)
        return self.head(torch.cat([oct_feat, fundus_feat], dim=1))


class TriModalLateFusionModel(_FusionBase):
    """Late concat of OCT + fundus + VF pooled features (GLEAM baseline)."""

    def __init__(
        self,
        oct_backbone: TokenBackbone,
        fundus_backbone: TokenBackbone,
        vf_backbone: TokenBackbone,
        num_classes: int = 4,
        embed_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__(oct_backbone, fundus_backbone, vf_backbone)
        self.oct_proj = _projection(oct_backbone.feat_dim, embed_dim)
        self.fundus_proj = _projection(fundus_backbone.feat_dim, embed_dim)
        self.vf_proj = _projection(vf_backbone.feat_dim, embed_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(3 * embed_dim),
            nn.Dropout(dropout),
            nn.Linear(3 * embed_dim, num_classes),
        )

    def forward(
        self,
        oct_image: torch.Tensor,
        fundus_image: torch.Tensor,
        vf_image: torch.Tensor,
    ) -> torch.Tensor:
        oct_feat = self.oct_proj(self.oct_backbone(oct_image)).mean(dim=1)
        fundus_feat = self.fundus_proj(self.fundus_backbone(fundus_image)).mean(dim=1)
        vf_feat = self.vf_proj(self.vf_backbone(vf_image)).mean(dim=1)
        return self.head(torch.cat([oct_feat, fundus_feat, vf_feat], dim=1))


class TriModalCrossAttentionFusionModel(_FusionBase):
    """Each modality attends over the concatenation of the other two."""

    def __init__(
        self,
        oct_backbone: TokenBackbone,
        fundus_backbone: TokenBackbone,
        vf_backbone: TokenBackbone,
        num_classes: int = 4,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__(oct_backbone, fundus_backbone, vf_backbone)
        self.oct_proj = _projection(oct_backbone.feat_dim, embed_dim)
        self.fundus_proj = _projection(fundus_backbone.feat_dim, embed_dim)
        self.vf_proj = _projection(vf_backbone.feat_dim, embed_dim)

        self.oct_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.fundus_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.vf_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.oct_norm = nn.LayerNorm(embed_dim)
        self.fundus_norm = nn.LayerNorm(embed_dim)
        self.vf_norm = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(3 * embed_dim),
            nn.Dropout(dropout),
            nn.Linear(3 * embed_dim, num_classes),
        )

    def forward(
        self,
        oct_image: torch.Tensor,
        fundus_image: torch.Tensor,
        vf_image: torch.Tensor,
    ) -> torch.Tensor:
        o = self.oct_proj(self.oct_backbone(oct_image))
        f = self.fundus_proj(self.fundus_backbone(fundus_image))
        v = self.vf_proj(self.vf_backbone(vf_image))

        o_ctx, _ = self.oct_attn(o, torch.cat([f, v], dim=1), torch.cat([f, v], dim=1))
        f_ctx, _ = self.fundus_attn(f, torch.cat([o, v], dim=1), torch.cat([o, v], dim=1))
        v_ctx, _ = self.vf_attn(v, torch.cat([o, f], dim=1), torch.cat([o, f], dim=1))

        pooled = torch.cat(
            [
                self.oct_norm(o + o_ctx).mean(dim=1),
                self.fundus_norm(f + f_ctx).mean(dim=1),
                self.vf_norm(v + v_ctx).mean(dim=1),
            ],
            dim=1,
        )
        return self.head(pooled)


def _projection(in_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim))


def _build_branch(
    branch_cfg: dict,
    modality: str,
    load_weights: bool,
) -> TokenBackbone:
    return TokenBackbone(
        *build_backbone(
            branch_cfg.get("name", "efficientnet_b0"),
            modality=modality,
            pretrained=bool(branch_cfg.get("pretrained", True)) and load_weights,
            checkpoint=branch_cfg.get("checkpoint") if load_weights else None,
        ),
        trainable=not bool(branch_cfg.get("freeze", False)),
    )


def build_fusion_model(
    cfg: dict,
    num_classes: int = 2,
    load_weights: bool = True,
) -> nn.Module:
    """Build a fusion model from the ``fusion:`` block of a training config."""
    oct_backbone = _build_branch(cfg.get("oct") or {}, "oct", load_weights)
    fundus_backbone = _build_branch(cfg.get("fundus") or {}, "fundus", load_weights)

    kind = str(cfg.get("type", "cross_attention")).lower().replace("-", "_")
    embed_dim = int(cfg.get("embed_dim", 512))
    dropout = float(cfg.get("dropout", 0.1))
    trimodal = bool(cfg.get("trimodal", False)) or kind.startswith("trimodal")

    if trimodal:
        vf_backbone = _build_branch(cfg.get("vf") or {}, "vf", load_weights)
        if kind in {"late", "late_fusion", "concat", "trimodal_late"}:
            return TriModalLateFusionModel(
                oct_backbone,
                fundus_backbone,
                vf_backbone,
                num_classes=num_classes,
                embed_dim=embed_dim,
                dropout=dropout,
            )
        return TriModalCrossAttentionFusionModel(
            oct_backbone,
            fundus_backbone,
            vf_backbone,
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_heads=int(cfg.get("num_heads", 8)),
            dropout=dropout,
        )

    if kind in {"late", "late_fusion", "concat"}:
        return LateFusionModel(
            oct_backbone,
            fundus_backbone,
            num_classes=num_classes,
            embed_dim=embed_dim,
            dropout=dropout,
        )
    if kind in {"cross_attention", "cross_attn", "xattn"}:
        return CrossAttentionFusionModel(
            oct_backbone,
            fundus_backbone,
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_heads=int(cfg.get("num_heads", 8)),
            dropout=dropout,
            bidirectional=bool(cfg.get("bidirectional", True)),
        )
    raise ValueError(f"Unknown fusion type: {cfg.get('type')}")
