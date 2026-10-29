# GLAUCO — Glaucoma Detection (OCT + Fundus)

Production-grade glaucoma classification extending the DRACO pipeline. Two independent single-modality specialists plus optional cross-attention fusion for paired data.

## Architecture decision (2026-08-29)

After structured debate across four architectural personas, see [GLAUCOMA_OVERVIEW.md](GLAUCOMA_OVERVIEW.md) / [GLAUCOMA_RESULTS_2026.md](GLAUCOMA_RESULTS_2026.md).

| Track | Model | Dataset | Target |
|-------|-------|---------|--------|
| OCT (RNFLT) | EfficientNet-B0 | Harvard-GF | AUC ≥ 0.90 |
| OCT (B-scan) | RETFound ViT-L | Harvard-GF | AUC ≥ 0.85 |
| Fundus | RETFound-CFP | AIROGS-light → full | AUC ≥ 0.92 |
| Fusion | Cross-attention | GAMMA → GLEAM | κ ≥ 0.85 |

**PINN rejected for v1** — no SOTA benchmark beats EfficientNet on Harvard-GDP; accuracy first.

## Latest research (2025–2026)

| Paper | Key finding |
|-------|-------------|
| **GLEAM / HAMM** (arXiv:2603.12800, TMI 2026) | First public tri-modal dataset (1,200 samples); hierarchical attentive fusion with modality gating beats dual-modal |
| **MAM-Fusion-Net** (Sci Reports 2026) | Multi-attention VF+CFP+OCT fusion; AUC 0.981 glaucoma vs non-glaucoma |
| **IMO** (ISBI 2026) | Diffusion-based iterative refinement for joint segmentation + grading |
| **Fairness LoRA** (arXiv:2512.03477) | MaxAccGap loss reduces accuracy disparity 69% on 10k fundus images |
| **Harvard-GDP benchmark** | EfficientNet AUC 0.90 on RNFLT; ViT only 0.75 at same data scale |

## Quick start

```powershell
# 1. Use Python 3.11 venv (PyTorch 3.14 on Windows is broken)
py -3.11 -m venv .venv
.\.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\pip install -r requirements.txt huggingface_hub matplotlib

# 2. Download smoke dataset (500 RNFLT maps, ~182 MB)
.\.venv\Scripts\python.exe scripts/download_glaucoma_datasets.py --dataset harvard_gd

# 3. Build catalogs
.\.venv\Scripts\python.exe scripts/build_glaucoma_catalogs.py --dataset harvard_gd

# 4. Train OCT/RNFLT baseline
.\.venv\Scripts\python.exe glauco/train.py --config configs/glaucoma_efficientnet_harvard_gd_smoke.yaml

# 5. Train RETFound RNFLT (needs checkpoints/weights/RETFound_oct.pth)
.\.venv\Scripts\python.exe glauco/train.py --config configs/glaucoma_retfound_rnflt_harvard_gd.yaml

# 6. Fundus track (download RETFound-CFP weights first)
set HF_TOKEN=hf_...
.\.venv\Scripts\python.exe scripts/download_retfound_weights.py --modality cfp
# Then download AIROGS-light from Kaggle and build catalog
```

## Dataset curriculum

1. **Smoke** — Harvard-GD (500 RNFLT maps) ✅ downloaded
2. **OCT primary** — Harvard-GF (3,300 patients, 20.4 GB Zenodo)
3. **Fundus primary** — AIROGS-light-v2 (Kaggle) → full AIROGS (54.9 GB)
4. **Fusion dev** — GAMMA (300 paired, Grand Challenge)
5. **Fusion publish** — GLEAM (1,200 tri-modal, Kaggle)
6. **External test** — ACRIMA, RIM-ONE DL, BEH, Drishti-GS (+ REFUGE/PAPILA/Chákṣu when available). See [PARTH_GLAUCOMA_FUNDUS.md](PARTH_GLAUCOMA_FUNDUS.md).

Registry: `data/glaucoma_registry.yaml`

## Project layout

```
glauco/
├── data/datasets.py      # CatalogDataset + PairedDataset (OCT+fundus fusion)
├── data/transforms.py    # Modality-aware augmentations
├── models/factory.py     # EfficientNet, ResNet, RETFound OCT/CFP
├── models/fusion.py      # CrossAttentionFusion + LateFusion baselines
├── train.py              # Single-modality + fusion training
└── evaluate.py           # AUC, sens@95%spec, external validation

configs/glaucoma_*.yaml     # Training configs per track
scripts/
├── download_glaucoma_datasets.py
└── build_glaucoma_catalogs.py
```

## Publication roadmap

1. **Phase 1 (now):** Reproduce Harvard-GDP benchmarks on RNFLT with EfficientNet
2. **Phase 2:** Fundus RETFound-CFP on AIROGS with REFUGE external validation
3. **Phase 3:** Cross-attention fusion on GAMMA, scale to GLEAM
4. **Phase 4 (novelty):** Equity metrics on Harvard-GF race-balanced cohort; fairness-aware fine-tuning
