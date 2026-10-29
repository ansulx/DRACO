# DRACO

**D**iabetic **R**etinopathy **A**nalysis via **C**ross-modal **O**CT

Multimodal AI for diabetic retinopathy screening — OCT-first pipeline with a separate fundus branch.

---

## Overview

| | |
|---|---|
| **Problem** | Automated detection and staging of diabetic retinopathy (DR) and diabetic macular edema (DME) from retinal imaging |
| **Modality (this repo)** | OCT B-scans — DME grading and DR-related structural analysis |
| **Companion** | Fundus pipeline (separate) for DR severity and lesion detection |
| **Status** | Phase 2 — DR/DME OCT pipeline complete; **Phase 3 — Glaucoma (OCT + fundus)** in progress |

---

## Documentation

| Doc | Description |
|-----|-------------|
| [docs/SETUP_OTHER_DEVICE.md](docs/SETUP_OTHER_DEVICE.md) | **Clone → env → data → weights** on a new machine |
| [docs/DATA.md](docs/DATA.md) | Datasets, labels, splits, and roles |
| [docs/PREPROCESSING.md](docs/PREPROCESSING.md) | Catalogs, transforms, label rules |
| [docs/TRAINING.md](docs/TRAINING.md) | Train / evaluate CLI and configs |
| [docs/RESULTS.md](docs/RESULTS.md) | Metrics and experiment log |
| [docs/GLAUCOMA.md](docs/GLAUCOMA.md) | **GLAUCO** — glaucoma OCT + fundus |
| [docs/GHOTA.md](docs/GHOTA.md) | **GHOTA** — DR 3-class OCT + fundus + fusion |
| [docs/GHOTA_RESULTS.md](docs/GHOTA_RESULTS.md) | GHOTA scores + trained fusion |
| [docs/PARTH_GLAUCOMA_FUNDUS.md](docs/PARTH_GLAUCOMA_FUNDUS.md) | External glaucoma fundus packs |
| [docs/MENTOR_FOUR_DATASETS.md](docs/MENTOR_FOUR_DATASETS.md) | Modality map for mentor / AIIMS |
| [docs/NOVELTY_AND_PRODUCTION.md](docs/NOVELTY_AND_PRODUCTION.md) | Novelty + production pipeline |
| [docs/SUPERVISOR_SHOWCASE.md](docs/SUPERVISOR_SHOWCASE.md) | Supervisor talking points |

---

## Repository layout

```
DRACO/
├── configs/            # YAML training configs (DRACO / GHOTA / GLAUCO)
├── docs/
├── draco/              # Original OCT DME package
├── ghota/              # DR Acc specialists + fusion + infer CLI
├── glauco/             # Glaucoma specialists + fusion
├── data/
│   ├── registry.yaml
│   ├── glaucoma_registry.yaml
│   ├── raw/            # Local datasets (gitignored)
│   └── processed/      # Catalog CSVs (gitignored — rebuild after download)
├── scripts/            # Download + catalog builders
├── checkpoints/        # Weights (gitignored)
└── requirements.txt
```

**Not in git:** raw datasets, catalogs, `.venv`, `checkpoints/*.pt`, Kaggle/HF secrets.  
**In git:** source, configs, docs, download/catalog scripts, small metrics JSON.

---

## Quick start (this machine)

```bash
pip install -r requirements.txt
python scripts/check_gpu.py
python scripts/build_catalogs.py
python draco/train.py --config configs/baseline_efficientnet_mmrdr.yaml
```

### New laptop / other device

See **[docs/SETUP_OTHER_DEVICE.md](docs/SETUP_OTHER_DEVICE.md)** — clone `glauco-ar-demo`, recreate venv, download data, rebuild catalogs, fetch RETFound weights (or copy `checkpoints/` separately).

### GHOTA (DR Acc)

```powershell
.\.venv\Scripts\python.exe ghota/train.py --config configs/ghota_oct_dme_retfound_sota.yaml
.\.venv\Scripts\python.exe ghota/train.py --config configs/ghota_fundus_dr_retfound_sota.yaml
.\.venv\Scripts\python.exe ghota/train_fusion.py --config configs/ghota_oefi_fusion_xattn.yaml
.\.venv\Scripts\python.exe -m ghota.infer --help
```

### GLAUCO (glaucoma)

See [docs/GLAUCOMA.md](docs/GLAUCOMA.md). Prefer Python 3.11 on Windows.

```powershell
.\.venv\Scripts\python.exe scripts/download_glaucoma_datasets.py --dataset harvard_gd
.\.venv\Scripts\python.exe scripts/build_glaucoma_catalogs.py --dataset harvard_gd
.\.venv\Scripts\python.exe glauco/train.py --config configs/glaucoma_efficientnet_harvard_gd_full.yaml
```

---

## Tracks

| Package | Task |
|---------|------|
| `draco` | 3-class DME on OCT (MMRDR) |
| `ghota` | DR Acc — OCT + fundus specialists + OEFI fusion |
| `glauco` | Glaucoma — RNFLT / fundus / paired fusion |

---

## Author

[ansulx](https://github.com/ansulx)

## License

TBD — dataset licenses apply to downstream use (MMRDR CC BY-NC-ND 4.0, etc.)
