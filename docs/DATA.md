# Data

Datasets used in DRACO for OCT-based diabetic retinopathy / DME analysis.

---

## Scope

- **In scope:** OCT B-scans, DME labels, DR-related OCT tasks
- **Out of scope (this repo):** Fundus images — handled by a separate pipeline
- **Fusion:** Fundus + OCT integration planned later

---

## Dataset inventory

| Dataset | OCT images | Labels | Status | Role |
|---------|------------|--------|--------|------|
| **MMRDR** | 2,938 | 3-class DME | Extract pending | Primary train / benchmark |
| **OEFI** | 1,113 | Binary DME + coarse DR | Ready | External validation |
| **OCTID** | 313 (107 DR + 206 Normal) | Disease category | Ready | Supplementary |
| **Kermany** | 109,309 | CNV / DME / DRUSEN / NORMAL | Extract optional | Scale (optional) |

---

## Label schemas

### MMRDR — 3-class DME (primary)

| Grade | Meaning |
|-------|---------|
| 0 | No DME |
| 1 | NCI DME — fluid outside central subfield |
| 2 | CI DME — center-involving, treatment-relevant |

Patient-level train/test split (~80/20). Devices: Zeiss Cirrus, Optovue RTVue-XR.

### OEFI — binary DME + coarse DR

| Field | Values |
|-------|--------|
| DME | 0 = no, 1 = yes |
| DR | 0, NPDR, PDR, or unknown (`-`) |

Multi-site, Mexico. **Use as external test only** — do not leak into MMRDR training evaluation.

### OCTID — disease class

| Class | Images |
|-------|--------|
| Diabetic Retinopathy | 107 |
| Normal | 206 |

No DME staging. Useful for DR vs normal and RETFound-style benchmarks.

### Kermany — 4-class maculopathy

| Class | Train | Test |
|-------|-------|------|
| DME | 11,348 | 250 |
| CNV | 37,205 | 250 |
| DRUSEN | 8,616 | 250 |
| NORMAL | 51,140 | 250 |

Multiple B-scans per patient — split by patient ID in filename.

---

## Recommended data strategy

```
Train  →  MMRDR-OCT (3-class DME)
Val    →  MMRDR-OCT official test split
Test   →  OEFI-OCT (binary DME generalization)
Extra  →  OCTID DR+Normal (optional fine-tune data)
```

Do **not** merge datasets without label harmonization (see [PREPROCESSING.md](PREPROCESSING.md)).

---

## Sources & citations

| Dataset | Link |
|---------|------|
| MMRDR | https://figshare.com/articles/dataset/MMRDR/29423747 |
| OEFI | https://github.com/Traslational-Visual-Health-Laboratory/OCT-AND-EYE-FUNDUS-DATASET |
| OCTID | https://doi.org/10.5683/SP/FLGZZE (DR), https://doi.org/10.5683/SP/WLW4ZT (Normal) |
| Kermany | https://data.mendeley.com/datasets/rscbjbr9sj/3 |

---

## Local layout

```
data/raw/
├── mmrdr/MMRDR-OCT/     # img/ + OCT.csv
├── oefi/OCT/ + OCT.csv
├── octid/dr/ + normal/
└── kermany/CellData/OCT/   # optional
```

Machine-readable catalog: [`data/registry.yaml`](../data/registry.yaml)
