# Training & Methodology

Model development for DRACO OCT branch (A4000 / CUDA).

---

## Objective

Beat or match 2026 SOTA on OCT-based DME classification, with strong external validation on OEFI.

---

## Tasks

| Task | Classes | Dataset | Metric |
|------|---------|---------|--------|
| **T1 — DME 3-class** | No / NCI / CI | MMRDR-OCT | Accuracy, macro-F1, per-class F1 |
| **T2 — DME binary** | No / Yes | MMRDR → OEFI | AUC, sensitivity @ 95% specificity |

---

## How to train

```bash
# Full A4000 run (EfficientNet)
python draco/train.py --config configs/baseline_efficientnet_mmrdr.yaml

# ResNet-50 (MMRDR paper baseline)
python draco/train.py --config configs/baseline_resnet50_mmrdr.yaml

# RETFound ViT-L (set checkpoint after HF access)
python draco/train.py --config configs/retfound_mmrdr.yaml

# Evaluate + OEFI external
python draco/evaluate.py --checkpoint checkpoints/efficientnet_b0_mmrdr/best.pt
```

Smoke configs (`*_smoke.yaml`) use a 240/116 stratified subset for CI / CPU debugging.

---

## Models

| Tier | Config | Notes |
|------|--------|-------|
| Baseline | `baseline_efficientnet_mmrdr.yaml` | A4000: batch 32, AMP |
| Baseline | `baseline_resnet50_mmrdr.yaml` | Align with MMRDR table |
| Foundation | `retfound_mmrdr.yaml` | ViT-L; batch 8; gated OCT weights optional |

### RETFound OCT weights

1. Request access: [RETFound_mae_natureOCT](https://huggingface.co/YukunZhou/RETFound_mae_natureOCT) or [monish563/RETFOUND](https://huggingface.co/monish563/RETFOUND)
2. `huggingface-cli login`
3. Download to `checkpoints/weights/RETFound_oct.pth`
4. Set `model.checkpoint` in `configs/retfound_mmrdr.yaml`

Without gated access the trainer uses ImageNet ViT-L as architecture stand-in (not true RETFound).

---

## Defaults (A4000 ~16 GB)

| Param | CNN | RETFound (official MAE recipe) |
|-------|-----|--------------------------------|
| Input | 224 | 224 |
| Batch | 32 | 8 |
| Optimizer | AdamW | AdamW + layer_decay 0.65 |
| LR | 1e-4 | blr 5e-3 → peak ≈ 1.56e-4 |
| Schedule | constant | cosine + 10-epoch warmup |
| Head warmup | — | 5 epochs (head only) |
| Epochs | 40 + early stop | 50 + early stop (patience 15) |
| Loss | Weighted CE | Weighted CE |
| AMP | yes | yes |

---

## Reproducibility

- Seed 42
- Config copied into `checkpoints/<run_name>/config.yaml`
- Metrics in `best_metrics.json` / `eval_report.json`
- Log summary in [RESULTS.md](RESULTS.md)
