# Preprocessing

How raw OCT data is prepared before training in DRACO.

---

## Pipeline overview

```
Raw OCT (JPEG)
    │
    ├─► Catalog index (path + label CSV) — no image duplication
    │
    ├─► Resize / crop → 224×224
    │
    ├─► Normalize → ImageNet mean/std
    │
    ├─► Label mapping → task schema
    │
    └─► Split enforcement → official MMRDR train/test; OEFI external only
```

---

## Actual implementation

| Script | Purpose |
|--------|---------|
| `scripts/extract_mmrdr_oct.py` | Merge Figshare parts and extract `MMRDR-OCT` only |
| `scripts/build_catalogs.py` | Write processed catalog CSVs + class weights |
| `draco/data/datasets.py` | PyTorch `CatalogDataset` + DataLoader |
| `draco/data/transforms.py` | Train / eval transforms |

### Output layout

```
data/processed/
  catalogs/
    mmrdr_train.csv
    mmrdr_test.csv
    oefi_external.csv
    octid_supplement.csv
  stats/
    class_weights.json
    image_stats.json
```

CSV columns: `path, label, label_name, split, dataset` (+ optional eye/dr).

---

## Image transforms

| Stage | Transform |
|-------|-----------|
| Train | Resize 256 → RandomCrop 224 → ColorJitter → ImageNet normalize |
| Eval | Resize 224 → CenterCrop 224 → ImageNet normalize |
| Flip | **Off** by default (OCT laterality) |

---

## Label rules

| Task | Source | Mapping |
|------|--------|---------|
| **T1 (primary)** | MMRDR | grade `0/1/2` as-is |
| **T2 (external)** | OEFI | binary DME `0/1`; 3-class softmax remapped as `P(DME)=P1+P2` |
| Class weights | MMRDR train | inverse frequency in `class_weights.json` |

---

## Rebuild catalogs

```bash
python scripts/extract_mmrdr_oct.py --keep-staging   # once
python scripts/build_catalogs.py
```
