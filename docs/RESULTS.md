# Results

Experiment results for DRACO. Updated after each benchmark run.

---

## Status

| Phase | Status |
|-------|--------|
| Data extract (MMRDR-OCT) | Done — 2,938 images |
| Catalogs | Done — train 2,376 / test 562 / OEFI 1,113 |
| Training stack | Done — `draco/train.py`, `evaluate.py` |
| EfficientNet smoke (CPU) | Done — 5 epochs on 240-image subset |
| RETFound smoke (CPU) | Done earlier — ImageNet stand-in only |
| EfficientNet full (A4000) | Done — 38 epochs, early stop @ epoch 28 |
| ResNet-50 full (A4000) | Done — 40 epochs, best @ epoch 38 |
| RETFound v1 (A4000) | Failed — flat lr=5e-4 collapsed |
| RETFound v2 (A4000) | Done — official recipe; F1 0.784 (beats paper) |

---

## Catalog stats

| Catalog | N |
|---------|---|
| `mmrdr_train.csv` | 2,376 |
| `mmrdr_test.csv` | 562 |
| `oefi_external.csv` | 1,113 |
| `octid_supplement.csv` | 313 |

---

## Target benchmarks (MMRDR-OCT, DME 3-class)

Published baselines from MMRDR paper:

| Model | ACC (DME) | F1 (DME) |
|-------|-----------|----------|
| RETFound | 0.897 | 0.759 |
| ResNet-50 | 0.890 | 0.701 |
| ViT | 0.883 | 0.700 |
| KeepFIT | 0.778 | 0.664 |
| FLAIR | 0.728 | 0.559 |

---

## Smoke results (CPU, subset — not comparable to paper)

Train: 240 images (80/class). Eval: smoke test 116 + full MMRDR test 562 + OEFI 1113.

### EfficientNet-B0 (`efficientnet_b0_mmrdr_smoke`)

| Split | ACC | Macro-F1 | AUC | Notes |
|-------|-----|----------|-----|-------|
| Smoke test (116) | 0.336 | 0.287 | — | 5 epochs |
| Full MMRDR test (562) | 0.187 | 0.201 | 0.680 (binary remap) | Undertrained |
| OEFI binary | 0.435 | F1 0.32 | **0.679** | Sens@95%Spec 0.0 |

Checkpoint: `checkpoints/efficientnet_b0_mmrdr_smoke/`

### RETFound / ViT-L stand-in (`retfound_mmrdr_smoke`)

1 epoch CPU smoke only (ImageNet ViT-L — **not** official RETFound OCT MAE weights).

| Split | ACC | Macro-F1 | AUC |
|-------|-----|----------|-----|
| Full MMRDR test | 0.064 | 0.040 | 0.458 (binary remap) |
| OEFI binary | 0.150 | F1 0.261 | 0.475 |

Expected: collapses until true OCT weights + full GPU training.

---

## Full A4000 leaderboard

```
Rank  Model          MMRDR-ACC  MMRDR-F1   OEFI-AUC   Date
----  -----          ---------  --------   --------   ----
  1   RETFound OCT v2  0.890    0.784      0.994      2026-07-08
  2   RETFound Focal   0.890    0.769      0.994      2026-07-20
  3   ResNet-50        0.835    0.707      0.968      2026-07-08
  4   EfficientNet-B0  0.794    0.678      0.761      2026-07-08
  -   RETFound OCT v1  0.064    0.040      0.668      2026-07-08  (collapsed)
```

### EfficientNet-B0 (`efficientnet_b0_mmrdr`) — A4000, full MMRDR

**Training:** 2,376 train / 562 test, batch 32, AMP, AdamW lr=1e-4, weighted CE. Early stopping at epoch 38 (best epoch 28).

| Split | ACC | Macro-F1 | Per-class F1 (0/1/2) | AUC | Notes |
|-------|-----|----------|----------------------|-----|-------|
| MMRDR test (562) | **0.794** | **0.678** | 0.853 / 0.326 / 0.855 | — | 3-class DME |
| OEFI binary (1,113) | 0.742 | F1 0.490 | — | **0.761** | Sens@95%Spec 0.006 |

Checkpoint: `checkpoints/efficientnet_b0_mmrdr/best.pt`  
Eval report: `checkpoints/efficientnet_b0_mmrdr/eval_report.json`

Class 1 (mild DME) remains the bottleneck — F1 0.326 vs 0.85+ for classes 0 and 2.

### ResNet-50 (`resnet50_mmrdr`) — A4000, full MMRDR

**Training:** same hyperparams as EfficientNet. 40 epochs, best @ epoch 38.

| Split | ACC | Macro-F1 | Per-class F1 (0/1/2) | AUC | Notes |
|-------|-----|----------|----------------------|-----|-------|
| MMRDR test (562) | **0.835** | **0.707** | 0.900 / 0.350 / 0.871 | — | Matches paper ResNet F1 (0.701) |
| OEFI binary (1,113) | 0.950 | F1 0.845 | — | **0.968** | Sens@95%Spec **0.934** |

Checkpoint: `checkpoints/resnet50_mmrdr/best.pt`  
Eval report: `checkpoints/resnet50_mmrdr/eval_report.json`

ResNet-50 beats EfficientNet on all metrics and matches the MMRDR paper ResNet-50 macro-F1 baseline.

### RETFound OCT v1 (`retfound_mmrdr`) — FAILED

Flat `lr=5e-4`, no layer decay / warmup → all class-1. Checkpoint not usable.

### RETFound OCT v2 (`retfound_mmrdr_v2`) — A4000, official MAE recipe

**Training:** blr=5e-3 → peak ≈1.56e-4, layer_decay=0.65, cosine + 10-epoch warmup, 5-epoch head warmup, weighted CE. Early stop @ epoch 34 (best @ 19).

| Split | ACC | Macro-F1 | Per-class F1 (0/1/2) | AUC | Notes |
|-------|-----|----------|----------------------|-----|-------|
| MMRDR test (562) | **0.890** | **0.784** | 0.929 / 0.506 / 0.916 | — | Beats paper F1 (0.759) |
| OEFI binary (1,113) | 0.970 | F1 0.904 | — | **0.994** | Sens@95%Spec **0.970** |

Checkpoint: `checkpoints/retfound_mmrdr_v2/best.pt`  
Eval report: `checkpoints/retfound_mmrdr_v2/eval_report.json`

Best DRACO model on MMRDR. Mild DME (class 1) improved to F1 0.51 vs ~0.33 on CNNs.

### RETFound OCT + Focal (`retfound_mmrdr_focal`) — A4000

Same recipe as v2; loss = class-weighted Focal (γ=2.0) + label smoothing 0.05. Early stop @ 48 (best @ 33).

| Split | ACC | Macro-F1 | Per-class F1 (0/1/2) | AUC | Notes |
|-------|-----|----------|----------------------|-----|-------|
| MMRDR test (562) | 0.890 | 0.769 | 0.932 / 0.462 / 0.914 | — | Below v2 F1 (0.784) |
| OEFI binary (1,113) | 0.976 | F1 0.922 | — | **0.994** | Sens@95%Spec **0.988** |

Checkpoint: `checkpoints/retfound_mmrdr_focal/best.pt`  

Focal did not beat weighted CE on MMRDR macro-F1; keep **v2** as primary OCT model. Focal is slightly stronger on OEFI sens@95%spec.

---

## Known blockers

- [x] MMRDR extracted to `data/raw/mmrdr/`
- [x] Preprocessing catalogs implemented
- [x] Train / evaluate CLI implemented
- [x] CUDA / A4000 visible to PyTorch (Studio Driver 610.62)
- [x] RETFound gated OCT weights downloaded
- [x] RETFound fine-tune recipe (official blr / LRD / warmup)
