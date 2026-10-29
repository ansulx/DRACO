# Parth glaucoma fundus wrangling (GLAUCO external)

Parth’s Oct 16 links — treated like GHOTA same-modality wrangling, but routed to **GLAUCO only** (binary glaucoma fundus / OD). **Do not mix into GHOTA** DR Acc catalogs.

| Dataset | Source | On disk | n (usable) | Role | Cool? |
|---------|--------|---------|------------|------|-------|
| **ACRIMA** | Kaggle `orvile/acrima…` (Downloads `archive.zip`) | `data/raw/glaucoma/acrima/` | 705 (396G / 309N) | External / OD-cropped fundus | **Yes — strongest plug-and-play** |
| **BEH** | Awesome-Medical BEH + Downloads zip | `data/raw/glaucoma/beh/` | 634 (171G / 463N) | External (site shift, Bangladesh) | **Yes — domain-shift test** |
| **RIM-ONE DL** | Kaggle `orvile/rim-one…` + GitHub miag-ull | `data/raw/glaucoma/rim_one/` | 485 by-hospital (172G / 313N) | External (official hospital split) | **Yes — classic domain-shift bench** |
| **BIOMISA** (Mendeley `trghs22fpg/4`) | Downloads `Dataset.zip` | `data/raw/glaucoma/biomisa/` | OD Color ~79 eyes / 44 pts; also Macula OCT+fundus | GLAUCO OD external; Macula = dual-use note only | **Cool for fusion later**; small OD n |
| **Drishti-GS1** | IIIT CVIT + Kaggle ONH mirror | `data/raw/glaucoma/drishti/` | 101 (70G / 31N; official 50/51 split) | External / ONH seg + Acc | **Nice but tiny**; good for demo/CDR story |

## Verdict (what’s cool for GLAUCO)

1. **Ship first:** ACRIMA + RIM-ONE (by-hospital) + BEH as fundus **external validation** after AIROGS/Harvard training.  
2. **Keep for later:** BIOMISA OD Color (patient-level) and its Macula OCT volumes if we ever want a tiny paired fundus↔OCT sandbox — still GLAUCO/research, not GHOTA DR labels.  
3. **Optional:** Drishti for ONH/CDR storytelling; too small to move Acc needles alone.

## Catalog commands

```powershell
.\.venv\Scripts\python.exe scripts/build_glaucoma_catalogs.py --dataset acrima
.\.venv\Scripts\python.exe scripts/build_glaucoma_catalogs.py --dataset beh
.\.venv\Scripts\python.exe scripts/build_glaucoma_catalogs.py --dataset rim_one
.\.venv\Scripts\python.exe scripts/build_glaucoma_catalogs.py --dataset biomisa_od
.\.venv\Scripts\python.exe scripts/build_glaucoma_catalogs.py --dataset drishti
```

Outputs: `data/processed/glaucoma/catalogs/{acrima,beh,rim_one,biomisa_od,drishti}_{train,test}.csv`

## GHOTA boundary

These are **glaucoma / OD / composite fundus–OCT** packs. They are **not** APTOS/IDRiD-style DR grading. Leave GHOTA Acc path unchanged.
