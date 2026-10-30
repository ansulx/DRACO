# External fundus sets — GLAUCO external experiments

**Model A (zero-shot):** `glauco_retfound_cfp_airogs_light` (AIROGS-light RETFound-CFP)  
**Model B (adaptation):** `glauco_retfound_cfp_acrima_ft` (A → fine-tune on ACRIMA train)

Artifacts: `results/glaucoma_external_external.json` · `results/glaucoma_external_acrima_ft_external.json`

---

## Protocol A — zero-shot external (Adopt as OOD test)

| Dataset | n | AUC | Acc | Sens | Spec | Adopt verdict |
|---------|--:|----:|----:|-----:|-----:|--------------|
| **Drishti full (India)** | 101 | **0.890** | 0.812 | 0.786 | 0.871 | **ADOPT** — Indian Acc story |
| Drishti official test | 51 | **0.951** | 0.863 | 0.816 | 1.000 | Adopt (small n) |
| **BEH** | 634 | **0.782** | 0.793 | 0.585 | 0.870 | **Adopt light** — site shift OK |
| ACRIMA full | 705 | 0.578 | 0.452 | 0.045 | 0.974 | Adopt for **FT**, not zero-shot |
| RIM-ONE hospital test | 174 | 0.567 | 0.690 | 0.071 | 0.983 | Adopt for **domain-shift bench** (hard) |

AIROGS in-domain holdout for reference: AUC **0.969**, Acc **0.919**.

**Why ACRIMA/RIM-ONE tank zero-shot:** both are **OD-cropped**; AIROGS is full-field fundus. Domain gap, not a labeling bug.

---

## Protocol B — fine-tune on ACRIMA (Adopt as adaptation data)

ACRIMA test after FT: **AUC 0.997 · Acc 0.972 · Sens@95spec 0.987**

| Transfer target | AUC | Acc | Note |
|-----------------|----:|----:|------|
| Drishti full | 0.886 | 0.802 | Still strong (India) |
| Drishti official test | 0.939 | 0.843 | Holds |
| BEH | 0.774 | 0.785 | Similar to zero-shot |
| RIM-ONE hospital test | 0.513 | 0.483 | Still fails — different sites/cameras |

---

## What to tell supervisor

1. **Adopt Drishti + BEH** as real external Acc for the AIROGS fundus specialist (India + Bangladesh).  
2. **Adopt ACRIMA** as an **adaptation / OD-crop** set — zero-shot fails, FT succeeds (AUC → 0.997).  
3. **Adopt RIM-ONE** as the **hard domain-shift benchmark** (by-hospital); do not expect Acc without multi-site training.  
4. **Do not Adopt any of these into GHOTA/DR.**

Commands:

```powershell
.\.venv\Scripts\python.exe scripts/eval_glauco_external_external.py
.\.venv\Scripts\python.exe glauco/train.py --config configs/glaucoma_retfound_cfp_acrima_ft.yaml
.\.venv\Scripts\python.exe glauco/evaluate.py --checkpoint checkpoints/glauco_retfound_cfp_acrima_ft/best.pt --modality fundus --test-csv data/processed/glaucoma/catalogs/drishti_full.csv
```
