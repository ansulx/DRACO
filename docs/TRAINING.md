# Training & Methodology

Model development plan for DRACO OCT branch.

---

## Objective

Beat or match **2026 SOTA** on OCT-based DME classification, with strong **external validation** on OEFI.

---

## Primary task

| Task | Classes | Dataset | Metric |
|------|---------|---------|--------|
| **T1 — DME 3-class** | No / NCI / CI | MMRDR-OCT | Accuracy, macro-F1, per-class F1 |
| **T2 — DME binary** | No / Yes | MMRDR → OEFI | AUC, sensitivity @ 95% specificity |

Secondary: DR coarse classification on OEFI (0 / NPDR / PDR).

---

## Models to benchmark

| Tier | Model | Type | Notes |
|------|-------|------|-------|
| Baseline | EfficientNet-B0 | CNN | Strong on small medical data |
| Baseline | ResNet-50 / ViT-L | ImageNet | MMRDR paper baselines |
| Foundation | RETFound (OCT weights) | MAE ViT | Nature 2023, OCT fine-tune |
| Foundation | OCTCube-M | 3D OCT | If volumes available |
| Foundation | KeepFIT V2 | Vision-language | OCT modality |
| Foundation | FLAIR | Vision-language | Weaker on OCT per MMRDR |

---

## Training protocol

### Phase A — Reproduce baselines (MMRDR official split)

1. Fine-tune each model on MMRDR-OCT train
2. Evaluate on MMRDR-OCT test
3. Compare to published MMRDR numbers (RETFound ~0.90 ACC on DME)

### Phase B — External validation

1. Best model from Phase A
2. Zero-shot or fine-tuned evaluation on OEFI-OCT (binary DME)
3. Report generalization gap (MMRDR test → OEFI test)

### Phase C — Ablation (optional)

- With vs without OCTID supplementary data
- Binary vs 3-class heads
- Input resolution (224 vs 512)

---

## Hyperparameters (starting point)

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR | 1e-4 to 5e-3 (model-dependent) |
| Batch size | 16–32 (GPU memory) |
| Epochs | 50 with early stopping |
| Loss | Cross-entropy (class weights for imbalance) |
| Seed | Fixed for reproducibility |

Foundation models: follow official fine-tune recipes (layer decay, linear probe vs full fine-tune).

---

## Class imbalance

MMRDR DME distribution is skewed toward CI-DME (~56%). Use:

- Weighted cross-entropy or focal loss
- Report per-class metrics, not accuracy alone
- Confusion matrix for NCI vs CI errors (clinical impact)

---

## Fundus integration (future)

Fundus pipeline runs separately. Planned fusion strategies:

- Late fusion (ensemble predictions)
- Cross-modal foundation model (OCTCube-IR style)

Not in current training scope.

---

## Reproducibility

- Fixed random seeds
- Log configs per run (YAML)
- Save checkpoints to `checkpoints/` (gitignored)
- Track experiments in `docs/RESULTS.md`
