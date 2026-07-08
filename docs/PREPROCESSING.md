# Preprocessing

How raw OCT data is prepared before training in DRACO.

---

## Pipeline overview

```
Raw OCT (JPEG)
    │
    ├─► Quality filter (optional — exclude unreadable scans)
    │
    ├─► Resize / pad → model input size (e.g. 512×512 or 224×224)
    │
    ├─► Normalize → ImageNet stats or dataset-specific mean/std
    │
    ├─► Label mapping → unified schema per task
    │
    └─► Split enforcement → patient-level, no leakage
```

---

## Image preprocessing

| Step | Detail |
|------|--------|
| **Format** | JPEG B-scans (MMRDR, OEFI, OCTID, Kermany) |
| **Resize** | Preserve aspect ratio or center-crop to square; match foundation model input (RETFound: 224 or 512) |
| **Normalization** | ImageNet mean/std for transfer learning; per-dataset stats optional for domain shift analysis |
| **Augmentation (train)** | Horizontal flip (careful with OCT orientation), slight brightness/contrast, optional random crop |

OCT B-scans are **not** naturally symmetric — augmentation policy will be validated against clinical convention.

---

## Label harmonization

Different datasets use incompatible labels. Map only when intentional:

### Binary DME (for cross-dataset experiments)

| Source | Positive (DME=1) | Negative (DME=0) |
|--------|------------------|------------------|
| MMRDR | grade 1 or 2 | grade 0 |
| OEFI | DME=1 | DME=0 |
| Kermany | class DME | CNV, DRUSEN, NORMAL |

### 3-class DME

Only **MMRDR** supports native 3-class labels. Do not force-map OEFI or Kermany.

### DR on OCT

Only **OEFI** has partial DR labels on OCT. Exclude `DR = '-'` for supervised DR tasks.

---

## Split rules

| Rule | Why |
|------|-----|
| **Patient-level splits** | Multiple slices per eye/patient in Kermany; MMRDR OCT split is patient-level |
| **No cross-dataset train+test overlap** | MMRDR train → OEFI test for honest generalization |
| **Official MMRDR split** | Use provided train/test for benchmark comparability |

---

## Output format (planned)

```
data/processed/
├── mmrdr_oct/
│   ├── train/
│   ├── test/
│   └── labels.csv
└── oefi_oct/
    └── test/
```

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/inventory_datasets.py` | Scan zips and extracted folders |
| `scripts/explore_local_data.py` | Label counts, image stats, missing files |
| `scripts/download_datasets.py` | Fetch MMRDR / OEFI (optional) |

Preprocessing scripts (`preprocess_*.py`) — **to be added** in Phase 2.
