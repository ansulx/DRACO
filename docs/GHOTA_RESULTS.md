# GHOTA Results — DR 3-class SOTA

_Generated from checkpoints on disk. Pack JSON: `docs/ghota/results_pack.json`._

## Summary

| Specialist | Holdout Acc | Macro-F1 | Checkpoint |
|------------|------------:|---------:|------------|
| **OCT** (MMRDR+OCTID) | **0.901** | **0.831** | `ghota_oct_dme_retfound_sota` |
| **Fundus** (APTOS) | **0.922** | **0.858** | `ghota_fundus_dr_retfound_sota` |

### OEFI external (binary DME, OCT model)

| AUC | Sens@0.5 | Spec@0.5 | F1 | Acc | Sens@95%Spec |
|----:|---------:|---------:|---:|----:|-------------:|
| 0.994 | 0.952 | 0.969 | 0.896 | 0.967 | 0.976 |

## Fusion status

**Paired OEFI is on disk now** (409 same-eye fundus+OCT pairs).

| Branch (binary DME) | AUC | Acc |
|---------------------|----:|----:|
| OCT specialist | **0.982** | 0.932 |
| Fundus specialist | 0.915 | 0.888 |
| Late fusion | 0.979 | 0.927 |

| Branch (3-class DR, n=409) | Acc | Macro-F1 |
|----------------------------|----:|---------:|
| OCT | 0.765 | 0.500 |
| Fundus | 0.756 | 0.446 |
| Late fusion | **0.780** | **0.511** |

Full write-up: [`TRAINED_FUSION.md`](TRAINED_FUSION.md) · mentor pack: [`MENTOR_FOUR_DATASETS.md`](MENTOR_FOUR_DATASETS.md)

### Trained fusion (patient holdout, n_test=101)

| Method | Acc | Macro-F1 |
|--------|----:|---------:|
| OCT / Fundus / Avg fuse | 0.703 / 0.653 / 0.733 | 0.485 / 0.372 / 0.527 |
| Feature MLP | 0.762 | 0.635 |
| **Cross-attn** | **0.802** | **0.714** |

Demo: `python -m ghota.infer --oct <o.jpg> --fundus <f.jpg> --fusion-ckpt checkpoints/ghota_oefi_fusion_xattn/best.pt`

## Figures

- `docs/ghota/figures/01_specialists_acc_f1.png`
- `docs/ghota/figures/02_oct_confusion.png`
- `docs/ghota/figures/03_fundus_confusion.png`
- `docs/ghota/figures/04_oefi_external.png`

## Recipe

RETFound (OCT / CFP) + Focal γ=2 + layer-decay 0.65 + cosine warmup + head warmup.

## One-liner

> GHOTA delivers strong DR specialists on OCT and fundus under one 3-class severity
> language, plus **trained cross-attn fusion on real OEFI same-eye pairs** (Acc 0.802).
> Glaucoma stays in GLAUCO.
