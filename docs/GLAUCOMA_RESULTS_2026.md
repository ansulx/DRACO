# GLAUCO — Full Results Pack (A4000)

**Hardware:** NVIDIA RTX A4000 16 GB (no H100 required).
**Goal:** Dataset inventory → 2026 literature comparison → trained models → graphs.

## 1. Datasets we used

| Dataset | Modality | N | Role | Link |
|---------|----------|---|------|------|
| Harvard-GD | RNFLT | 500 | OCT smoke / high AUC | [HF Harvard-GD](https://huggingface.co/datasets/harvardairobotics/Harvard-GD) |
| Harvard-GDP | RNFLT | 1,000 | OCT scale | [HF Harvard-GDP](https://huggingface.co/datasets/harvardairobotics/Harvard-GDP) |
| Harvard-GF | RNFLT (+ OCT) | 3,300 | Primary OCT benchmark | [Zenodo](https://zenodo.org/records/13178701) |
| AIROGS-light-v2 | Fundus | ~9.5k | Primary fundus | [Kaggle](https://www.kaggle.com/datasets/deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2) |
| GLEAM | SLO + cpOCT + VF | 1,200 | Fusion / staging (2026) | [Kaggle](https://www.kaggle.com/datasets/zhangyiyinge/gleam-dataset) |

Optional external (not required for this pack): [REFUGE](https://ieee-dataport.org/documents/refuge-retinal-fundus-glaucoma-challenge), [GAMMA](https://gamma.grand-challenge.org/).

## 2. 2026 / recent papers we compare against

| Paper | What to match | Link |
|-------|---------------|------|
| **HAMM / GLEAM (TMI 2026)** | Tri-modal Acc≈81%, F1≈76%, AUROC≈93% | [arXiv:2603.12800](https://arxiv.org/abs/2603.12800) · [GitHub](https://github.com/microewing/HAMM) |
| **M2F3 (TVST)** | Dual fusion κ≈0.90 on GAMMA | [DOI](https://doi.org/10.1167/tvst.14.11.8) |
| **RETFound (Nature 2023)** | Fundus/OCT foundation backbone | [Nature](https://www.nature.com/articles/s41586-023-06555-x) |
| Harvard RNFLT EfficientNet | OCT AUC ≈ 0.90 | [Harvard-GF](https://github.com/Harvard-Ophthalmology-AI-Lab/Harvard-GF) |

## 3. Our specialist results (screening)

| Model | Dataset | Metric |
|-------|---------|--------|
| OCT/RNFLT · Harvard-GF | `glauco_efficientnet_harvard_gf_full` | AUC 0.838, F1 0.772 |
| Fundus · AIROGS-light | `glauco_retfound_cfp_airogs_light` | AUC 0.983, F1 0.936 |
| OCT/RNFLT · Harvard-GD | `glauco_efficientnet_harvard_gd_full` | AUC 0.913, F1 0.830 |
| Dual fusion (RETFound+Eff, prior) | `glauco_fusion_gleam_crossattn` | F1 0.506 |

## 4. GLEAM fusion results (test set)

| Run | Acc | Macro-F1 | Notes |
|-----|-----|----------|-------|
| `glauco_fusion_gleam_trimodal_xattn` | 0.742 | 0.689 | binary AUC 0.947 |
| `glauco_fusion_gleam_trimodal_late` | 0.762 | 0.704 | binary AUC 0.947 |
| `glauco_fusion_gleam_dual_xattn_eff` | 0.525 | 0.492 | binary AUC 0.801 |
| `glauco_fusion_gleam_late` | 0.517 | 0.479 | binary AUC 0.814 |
| `glauco_fusion_gleam_binary_xattn` | 0.725 | — | AUC 0.834 |
| `glauco_fusion_gleam_crossattn` | 0.554 | 0.471 | binary AUC 0.805 |

**Literature bar (HAMM tri-modal):** Acc 0.811, F1 0.759, AUROC 0.930.

## 5. Figures

- `results/glaucoma_figures/gleam_vs_hamm.png`
- `results/glaucoma_figures/specialist_auc.png`
- `results/glaucoma_figures/trimodal_training_curve.png`

## 6. How to demo / retrain

```powershell
cd C:\Users\HP\OneDrive\Desktop\OCT
.\.venv\Scripts\python.exe scripts\run_glaucoma_paper_pack.py
.\.venv\Scripts\python.exe scripts\glauco_infer.py --help
```

Machine-readable pack: `results/glaucoma_paper_pack.json`.

## 7. Long improvement pack (auto)

Finished: 2026-09-06T16:43

| Run | Acc | Macro-F1 | AUC / binary AUC |
|-----|-----|----------|------------------|
| `glauco_gleam_vf_eff` | 0.742 | 0.672 | 0.884 |
| `glauco_gleam_fundus_eff` | 0.558 | 0.449 | 0.803 |
| `glauco_gleam_oct_eff` | 0.537 | 0.489 | 0.757 |
| `glauco_fusion_gleam_trimodal_binary` | 0.867 | — | 0.955 |
| `glauco_fusion_gleam_trimodal_xattn_v2` | 0.742 | 0.689 | 0.948 |
| `glauco_fusion_gleam_trimodal_xattn` | 0.742 | 0.689 | 0.947 |
| `glauco_fusion_gleam_trimodal_late` | 0.762 | 0.704 | 0.947 |
| `glauco_fusion_gleam_dual_xattn_eff` | 0.525 | 0.492 | 0.801 |

Figure: `results/glaucoma_figures/long_pack_gleam_vs_hamm.png`
