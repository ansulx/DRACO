# GLAUCO — Master Overview

**Project:** Glaucoma detection (OCT + fundus), Phase 3 after DRACO (DR/DME on OCT).  
**Hardware:** RTX A4000 16 GB — always use `.venv` (Python 3.11 + PyTorch 2.6+cu124).  
**Priority:** Accuracy at scale. Novelty (PINN, fairness) later.

---

## 1. The whole setup (one picture)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         YOUR MACHINE (A4000)                            │
├─────────────────────────────────────────────────────────────────────────┤
│  DRACO (done)          │  GLAUCO (building)                              │
│  ─────────────         │  ─────────────────                               │
│  MMRDR  2,938 OCT      │  Harvard-GD   500 RNFLT    ✓ trained AUC 0.913  │
│  OEFI   1,114 OCT      │  Harvard-GDP 1,000 RNFLT   ✓ catalog ready      │
│  RETFound_oct ✓        │  Harvard-GF  3,300 RNFLT   ⏳ download next       │
│                        │  AIROGS     113,893 fundus ⏳ largest fundus set  │
│                        │  GAMMA/GLEAM  paired fusion ⏳ paper track        │
├─────────────────────────────────────────────────────────────────────────┤
│  glauco/train.py  →  catalog CSV  →  model checkpoint  →  evaluate.py   │
│  Models: EfficientNet (RNFLT) | RETFound OCT (B-scan) | RETFound-CFP    │
│          CrossAttentionFusion (paired OCT + fundus)                       │
└─────────────────────────────────────────────────────────────────────────┘
```

**Flow:** `download` → `build_glaucoma_catalogs.py` → `glauco/train.py` → `glauco/evaluate.py`  
**Pattern:** Same as DRACO — images stay on disk, CSV catalogs are the only training interface.

---

## 2. Full dataset landscape (everything worth using)

### OCT / RNFLT — structure maps (your best track so far)

| Dataset | Patients/Images | On disk? | Download | Role |
|---------|-----------------|----------|----------|------|
| **Harvard-GF** | **3,300** | No | [Zenodo 20.4 GB](https://zenodo.org/records/13178701) | **Primary train** — largest public RNFLT+OCT, official splits, race-balanced |
| **Harvard-GDP** | 1,000 | **Yes (20 GB)** | HF (done) | External val + progression labels |
| **Harvard-GD** | 500 | **Yes (0.5 GB)** | HF (done) | Smoke test — already hit AUC 0.913 |

Harvard-GF is the OCT scale-up target. It is 6.6× larger than GD and has both RNFLT maps and 3D B-scans per patient.

### OCT / B-scan — volumetric (RETFound territory)

| Dataset | Size | In GF/GDP? | Model |
|---------|------|------------|-------|
| Harvard-GF B-scans | 3,300 × 200³ voxels | Inside GF zip | RETFound OCT (DRACO v2 recipe) |
| Harvard-GDP B-scans | 1,000 `.npz` | **On disk** | Same |
| GAMMA macular OCT | 300 volumes | Separate | Fusion branch only |

### Fundus — color disc photos (largest public glaucoma data)

| Dataset | Images | On disk? | Download | Role |
|---------|--------|----------|----------|------|
| **AIROGS full** | **113,893** | No | [Zenodo 54.9 GB](https://zenodo.org/records/5793241) (approval) | **Primary train** at scale |
| **AIROGS-light-v2** | ~8,000 balanced | No | [Kaggle](https://www.kaggle.com/datasets/deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2) | Fast iteration first |
| SMDG-19 | 12,449 (19 sources) | No | Kaggle/HF | Multi-source training |
| REFUGE | 1,200 | No | IEEE DataPort | **External test only** |
| PAPILA | 488 | No | Sci Data | External test |
| Chákṣu | 1,345 | No | Sci Data | External test |
| RIM-ONE DL | 485 | No | GitHub | External test (by-hospital split) |

Fundus at “great” scale = **AIROGS full (113k)**. Start with AIROGS-light (~8k) to validate pipeline, then scale.

### Paired / fusion (OCT + fundus together)

| Dataset | Pairs | Modalities | On disk? | Role |
|---------|-------|------------|----------|------|
| **GLEAM** | 1,200 | Fundus + cpOCT + VF, 4-stage | No | **2026 paper target** |
| **GAMMA** | 300 | Fundus + macular OCT, 3-class | No | Fusion dev / benchmark κ≥0.85 |
| Harvard-GDP/GF | — | OCT only (no fundus) | Partial | Not paired |

No dataset pairs fundus + OCT at AIROGS scale. Fusion papers use GAMMA (300) then GLEAM (1,200).

### DRACO datasets (already on disk — not for glaucoma labels)

| Dataset | Size | Glaucoma use? |
|---------|------|---------------|
| MMRDR | 36.6 GB, 2,938 OCT | **No** — DME labels only |
| OEFI | 0.2 GB, 1,114 OCT | **No** — DR labels; fundus folder skipped |

Do not repurpose DR/OEFI for glaucoma — labels are wrong disease.

---

## 3. What you have on disk today (Aug 31, 2026)

| Asset | Size | Status |
|-------|------|--------|
| `data/raw/glaucoma/harvard_gd` | 0.52 GB | Trained — AUC 0.913 |
| `data/raw/glaucoma/harvard_gdp` | 20.44 GB | Catalog: 800 train / 200 test |
| `data/raw/mmrdr` | 36.56 GB | DRACO only |
| `data/raw/oefi` | 0.21 GB | DRACO only |
| `checkpoints/weights/RETFound_oct.pth` | ~3.8 GB | Ready |
| `checkpoints/weights/RETFound_cfp.pth` | — | **Not downloaded** — needed for fundus |
| Catalogs | 4 CSVs | GD + GDP RNFLT |
| Glauco checkpoints | 3 runs | EfficientNet ×2, RETFound ×1 |

---

## 4. Models per modality (locked in)

| Modality | Model | Why | Proven AUC |
|----------|-------|-----|------------|
| RNFLT heatmaps | **EfficientNet-B0** | Beats ViT on thickness maps | **0.913** (GD) |
| OCT B-scans | **RETFound ViT-L** | MAE pretrain, DRACO recipe works | 0.994 OEFI (DR task) |
| Fundus | **RETFound-CFP** | Best external generalization | target 0.92 |
| Paired fusion | **CrossAttentionFusion** | +0.19 κ over single-modality (GAMMA lit.) | target κ 0.85 |

PINN: rejected for v1.

---

## 5. Scale-up plan (train on “great” data)

### Phase A — OCT at real scale (do this first)

1. Download **Harvard-GF** from Zenodo (~20 GB) → `data/raw/glaucoma/harvard_gf`
2. Add `build_harvard_gf` to catalog script (RNFLT + optional B-scan paths)
3. Train EfficientNet on **2,100 official train** → validate on **900 official test**
4. Keep **Harvard-GDP** as untouched external (train GF, never touch GDP train split for final numbers)
5. Optional: RETFound on GF B-scans (second OCT specialist)

**Expected:** AUC ~0.90–0.93 on RNFLT (literature benchmark). Equity metrics free on GF (race-balanced).

### Phase B — Fundus at scale

1. `download_retfound_weights.py --modality cfp` (needs HF token)
2. **AIROGS-light-v2** from Kaggle (~few GB) → pipeline check
3. **AIROGS full** from Zenodo (54.9 GB, request access) → production fundus model
4. Hold out **REFUGE + PAPILA + Chákṣu** for external AUC (never train on these)

**Expected:** Internal AUC 0.92+, external 0.89–0.93 (realistic, not leaderboard 0.99).

### Phase C — Fusion (publication)

1. Register **GAMMA** → train `configs/glaucoma_fusion_gamma_crossattn.yaml`
2. Download **GLEAM** (Kaggle) → tri-modal paper result
3. Compare CrossAttention vs LateFusion ablation

### Phase D — Novelty (after accuracy locked)

- Fairness on Harvard-GF demographics
- Progression on Harvard-GDP `progression` labels (500 patients)
- OD crop / CDR auxiliary head on fundus

---

## 6. A4000 training settings (what fits)

| Track | Batch | Model | VRAM | Notes |
|-------|-------|-------|------|-------|
| RNFLT EfficientNet | 32 | ~50 MB | Easy | Scale to GF with batch 32 |
| RNFLT RETFound | 8 | ~4 GB | OK | layer_decay + blr 5e-3 |
| Fundus RETFound-CFP | 16 | ~4 GB | OK | stronger aug, balance_classes |
| Fusion cross-attn | 4 | ~8 GB | Tight | freeze backbones first |

Full AIROGS training: days not hours, but A4000 handles it fine with AMP.

---

## 7. Commands cheat sheet

```powershell
cd C:\Users\HP\OneDrive\Desktop\OCT
.\.venv\Scripts\activate

# OCT — works today on GD
python scripts/build_glaucoma_catalogs.py --dataset harvard_gd
python glauco/train.py --config configs/glaucoma_efficientnet_harvard_gd_full.yaml

# OCT — external val on GDP (after training on GD or GF)
python glauco/evaluate.py --checkpoint checkpoints/glauco_efficientnet_harvard_gd_full/best.pt `
  --test-csv data/processed/glaucoma/catalogs/harvard_gdp_test.csv --modality rnflt

# Scale downloads
python scripts/download_glaucoma_datasets.py --dataset harvard_gf   # prints Zenodo link
python scripts/download_glaucoma_datasets.py --dataset airogs_light # needs Kaggle API

# Weights
set HF_TOKEN=hf_...
python scripts/download_retfound_weights.py --modality both
```

---

## 8. Results so far

| Run | Data | AUC | Notes |
|-----|------|-----|-------|
| EfficientNet full | Harvard-GD (500) | **0.913** | Exceeds 0.90 benchmark |
| EfficientNet smoke | Harvard-GD (500) | 0.901 | Meets benchmark |
| RETFound RNFLT | Harvard-GD (500) | 0.883 | CNN wins on heatmaps |

Index: `results/glaucoma_training_index.json`

---

## 9. Repo map

```
glauco/                    # Glaucoma package
configs/glaucoma_*.yaml    # 5 training configs
scripts/download_glaucoma_datasets.py
scripts/build_glaucoma_catalogs.py
data/glaucoma_registry.yaml
docs/GLAUCOMA*.md
draco/                     # DR/DME (separate task, reuse optim/losses/metrics)
```

---

## 10. Bottom line

You have validated the **pipeline on 500 samples** (AUC 0.913). For training on something **large and publication-grade**:

| Priority | Dataset | Size | Modality |
|----------|---------|------|----------|
| **#1** | Harvard-GF | 3,300 | OCT/RNFLT + B-scan |
| **#2** | AIROGS full | 113,893 | Fundus |
| **#3** | GLEAM | 1,200 | Fusion paper |

Harvard-GDP (1,000, already on disk) is your **external OCT validator** — don’t train final models on it if you use it for reporting.

Next concrete step: **download Harvard-GF from Zenodo** and we wire the catalog builder for its layout.

---

## 11. Autopilot (unattended training)

Leave the machine on and run:

```powershell
.\.venv\Scripts\python.exe scripts/run_glaucoma_autopilot.py
```

**Monitor while away:**
```powershell
.\scripts\check_autopilot.ps1
# or
Get-Content results\autopilot_status.json
```

**What it runs automatically:**
1. Rebuild catalogs (GD, GDP, combined 1,200 train)
2. Train EfficientNet on **Harvard-GDP** (800 train)
3. Train EfficientNet on **combined GD+GDP** (1,200 train)
4. Cross-evaluate: GDP model → GD test, combined → GDP/GD tests
5. Download RETFound-CFP if `HF_TOKEN` is set

**Logs:** `results/autopilot_logs/autopilot_*.log`  
**Live status:** `results/autopilot_status.json`
