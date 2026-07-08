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
| **Status** | Phase 1 — data collection & audit |

---

## Documentation

| Doc | Description |
|-----|-------------|
| [docs/DATA.md](docs/DATA.md) | Datasets, labels, splits, and roles |
| [docs/PREPROCESSING.md](docs/PREPROCESSING.md) | Image normalization, splits, and label harmonization |
| [docs/TRAINING.md](docs/TRAINING.md) | Model methodology and benchmark plan |
| [docs/RESULTS.md](docs/RESULTS.md) | Metrics and experiment results (updated per run) |

---

## Repository layout

```
DRACO/
├── docs/           # Project documentation (display)
├── data/
│   ├── registry.yaml
│   └── raw/        # Local datasets (not in git)
├── scripts/        # Download, inventory, exploration
└── requirements.txt
```

---

## Quick start

```bash
pip install -r requirements.txt
python scripts/inventory_datasets.py    # audit local data
python scripts/explore_local_data.py    # label distributions
```

---

## Primary task

**3-class DME on OCT** — No DME / NCI DME / CI DME (MMRDR benchmark)

---

## Author

[ansulx](https://github.com/ansulx)

## License

TBD — dataset licenses apply to downstream use (MMRDR CC BY-NC-ND 4.0, etc.)
