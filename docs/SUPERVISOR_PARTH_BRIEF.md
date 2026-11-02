# Supervisor pack — external fundus datasets (get · test · say)

**Primary deck:** `GLAUCO_External_Fundus_Datasets.pptx` (repo root + `docs/`) — 14-slide story arc  
**Full speaking script:** `docs/GLAUCO_DECK_SCRIPT.md` ← use this in the meeting  
**Also:** `docs/GLAUCO_EDGE_METRICS.md` · `docs/GLAUCO_DATASETS_USED.md`  
**Rebuild deck:** `python scripts/build_glauco_external_fundus_pptx.py`

---

## A. What you already have (no more download)

| Dataset | On disk | Catalogs | Experimented? |
|---------|---------|----------|---------------|
| ACRIMA | ✓ `data/raw/glaucoma/acrima/` | ✓ + external | ✓ zero-shot + FT |
| BEH | ✓ `data/raw/glaucoma/beh/` | ✓ + external | ✓ zero-shot |
| RIM-ONE DL | ✓ `data/raw/glaucoma/rim_one/` | ✓ hospital test | ✓ zero-shot + after FT |
| Drishti-GS1 | ✓ `data/raw/glaucoma/drishti/` | ✓ full + official test | ✓ zero-shot + after FT |
| BIOMISA | ✓ `data/raw/glaucoma/biomisa/` | OD Color only | Cataloged; **not** main Acc yet |

**You do NOT need the IIIT Drishti registration form** — Kaggle mirror already has labeled Images.

---

## B. What you still need to *get* (optional / next)

| Item | Why | Action |
|------|-----|--------|
| **OCTA-500** access | Earlier Parth DR/OCTA link — new modality | IEEE DataPort + email author |
| **ROAD** access | WF-OCTA with DR grades | Email authors |
| Copy of `checkpoints/` to other PC | Weights not in git | USB / Drive |
| Kaggle + HF tokens on other PC | Re-download if needed | Local secrets only |

**Nothing else required for tomorrow’s GLAUCO fundus / Indian-dataset talk.**

---

## C. What to *test* / show supervisors (checklist)

Open the PPT. Then if they ask for a live demo:

```powershell
# 1) Re-run zero-shot table (AIROGS → Parth externals)
.\.venv\Scripts\python.exe scripts/eval_glauco_parth_external.py

# 2) Show ACRIMA-adapted model on Indian Drishti
.\.venv\Scripts\python.exe glauco/evaluate.py `
  --checkpoint checkpoints/glauco_retfound_cfp_acrima_ft/best.pt `
  --modality fundus `
  --test-csv data/processed/glaucoma/catalogs/drishti_full.csv `
  --out results/demo_drishti_from_acrima_ft.json
```

**Numbers to point at (memorize these four):**

| Story | Number |
|-------|--------|
| In-domain AIROGS fundus | AUC **0.97** |
| **Indian** Drishti external | AUC **0.89** (full) / **0.95** (official test) |
| ACRIMA after we fine-tune | AUC **0.997** |
| ACRIMA / RIM-ONE *without* FT (OD-crop gap) | AUC ~**0.57** ← honest failure |

---

## D. Hire / pass card (hand this mental card)

| Dataset | Disease | Modality | India? | **Hire?** | As what |
|---------|---------|----------|--------|-----------|---------|
| **Drishti-GS1** | Glaucoma | Fundus + OD/cup | **YES (Aravind)** | **YES** | Indian external Acc |
| **BEH** | Glaucoma | Fundus | No (BD) | **YES (light)** | South-Asia site shift |
| **ACRIMA** | Glaucoma | Fundus OD-crop | No (ES) | **YES** | Adaptation / FT data |
| **RIM-ONE DL** | Glaucoma | Fundus OD-crop | No (ES) | **YES** | Hard hospital OOD bench |
| **BIOMISA** | Glaucoma + macula | Fundus + OCT | No (PK) | **Later** | Tiny OD n; optional fusion |
| Any of above → GHOTA/DR | — | — | — | **NO** | Wrong disease / labels |

Indian **DR** (already separate): **IDRiD + APTOS** on GHOTA — don’t confuse with these five.

---

## E. Speaking script

**Use the full deck script:** [`docs/GLAUCO_DECK_SCRIPT.md`](GLAUCO_DECK_SCRIPT.md)  
Order: disease → datasets → classes → problem → methods → Sens/Spec results → graphs → peak packs → ask.

**Ultra-short fallback (~90 seconds) if time is cut:**

These are **glaucoma** sets under **GLAUCO**, not DR. Peak fundus Acc is **AIROGS** (AUC **0.97**); peak OCT is **Harvard-GD** (AUC **0.91**); both **2-class**. Externals **Drishti / BEH / ACRIMA / RIM-ONE** are test-only — Drishti (India) transfers at **0.89–0.95**; OD-crop zero-shot fails (~**0.57**) until ACRIMA FT (**0.997**). Ask: keep binary edge packs; externals stay test-only.

---

## F. If they ask follow-ups

| Question | Answer |
|----------|--------|
| Why not train GHOTA on BIOMISA DME? | Labels are disease folders, not ICDR 0–4 grades like APTOS/IDRiD. |
| Why is Drishti only 101? | Classic Indian ONH benchmark — good for external Acc + CDR story, not for primary training scale. |
| Do we need IIIT register? | No — we already have labeled images on disk. |
| What’s next? | Keep reporting Drishti/BEH as standing externals; optional multi-site FT including RIM-ONE train hospitals; OCTA-500/ROAD when access arrives. |

---

## G. Files to open in the meeting

1. `SUPERVISOR_Showcase.pptx` (slides 5b + 5c)  
2. `docs/PARTH_DATASET_SPECS.md` (modality / India table)  
3. `docs/PARTH_EXTERNAL_RESULTS.md` (full metrics)  
4. This brief: `docs/SUPERVISOR_PARTH_BRIEF.md`
