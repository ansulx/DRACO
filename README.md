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
| **Status** | Phase 2 — preprocessing + training pipeline (A4000) |

---

## Documentation

| Doc | Description |
|-----|-------------|
| [docs/DATA.md](docs/DATA.md) | Datasets, labels, splits, and roles |
| [docs/PREPROCESSING.md](docs/PREPROCESSING.md) | Catalogs, transforms, label rules |
| [docs/TRAINING.md](docs/TRAINING.md) | Train / evaluate CLI and configs |
| [docs/RESULTS.md](docs/RESULTS.md) | Metrics and experiment log |

---

## Repository layout

```
DRACO/
├── configs/            # YAML training configs
├── docs/
├── draco/              # Python package (data, models, train, evaluate)
├── data/
│   ├── registry.yaml
│   ├── raw/            # Local datasets (gitignored)
│   └── processed/      # Catalog CSVs (gitignored)
├── scripts/            # Extract, inventory, build_catalogs
├── checkpoints/        # Run outputs (gitignored)
└── requirements.txt
```

---

## Quick start

```bash
pip install -r requirements.txt

# 0) GPU must work (A4000) — see docs/GPU_SETUP.md
python scripts/check_gpu.py

# 1) Extract MMRDR-OCT (once)
python scripts/extract_mmrdr_oct.py --keep-staging

# 2) Build catalogs
python scripts/build_catalogs.py

# 3) Train EfficientNet-B0 on A4000
python draco/train.py --config configs/baseline_efficientnet_mmrdr.yaml

# 4) External validation on OEFI
python draco/evaluate.py --checkpoint checkpoints/efficientnet_b0_mmrdr/best.pt
```

---

## Primary task

**3-class DME on OCT** — No DME / NCI DME / CI DME (MMRDR benchmark)

---

## Author

[ansulx](https://github.com/ansulx)

## License

TBD — dataset licenses apply to downstream use (MMRDR CC BY-NC-ND 4.0, etc.)
