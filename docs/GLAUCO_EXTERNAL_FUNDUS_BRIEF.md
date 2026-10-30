# Supervisor pack — external datasets (get · test · say)

**Primary deck:** `GLAUCO_External_Fundus_Datasets.pptx` (repo root + `docs/`) — 9-slide project briefing  
**Also in main showcase:** `SUPERVISOR_Showcase.pptx` → slides **5b / 5c**  
**This file:** get · test · short script  
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
| **OCTA-500** access | Earlier external DR/OCTA link — new modality | IEEE DataPort + email author |
| **ROAD** access | WF-OCTA with DR grades | Email authors |
| Copy of `checkpoints/` to other PC | Weights not in git | USB / Drive |
| Kaggle + HF tokens on other PC | Re-download if needed | Local secrets only |

**Nothing else required for tomorrow’s GLAUCO fundus / Indian-dataset talk.**

---

## C. What to *test* / show supervisors (checklist)

Open the PPT. Then if they ask for a live demo:

```powershell
# 1) Re-run zero-shot table (AIROGS → external externals)
.\.venv\Scripts\python.exe scripts/eval_glauco_external_external.py

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

## D. Adopt / pass card (hand this mental card)

| Dataset | Disease | Modality | India? | **ADOPT?** | As what |
|---------|---------|----------|--------|-----------|---------|
| **Drishti-GS1** | Glaucoma | Fundus + OD/cup | **YES (Aravind)** | **YES** | Indian external Acc |
| **BEH** | Glaucoma | Fundus | No (BD) | **YES (light)** | South-Asia site shift |
| **ACRIMA** | Glaucoma | Fundus OD-crop | No (ES) | **YES** | Adaptation / FT data |
| **RIM-ONE DL** | Glaucoma | Fundus OD-crop | No (ES) | **YES** | Hard hospital OOD bench |
| **BIOMISA** | Glaucoma + macula | Fundus + OCT | No (PK) | **Later** | Tiny OD n; optional fusion |
| Any of above → GHOTA/DR | — | — | — | **NO** | Wrong disease / labels |

Indian **DR** (already separate): **IDRiD + APTOS** on GHOTA — don’t confuse with these five.

---

## E. Short script — “what I analyzed through the datasets” (~2–3 min)

*Say this almost as written. Pause on bold.*

---

Sir / Ma’am — We reviewed five recent fundus links. I checked **disease, modality, geography, and whether we should Adopt them**.

**First finding — disease:**  
All five are **glaucoma** (or glaucoma plus other macular disease). They are **not** diabetic retinopathy Acc packs. So they go under **GLAUCO**, not GHOTA. Our DR path stays on **APTOS / IDRiD / OEFI**.

**Second — modalities:**  
Four are **colour fundus** only. **BIOMISA** is fundus **plus OCT**, but the glaucoma OD part is small (~44 patients), so we keep it for later, not as the Acc headline.

**Third — India:**  
In this batch, only **Drishti-GS1** is an **Indian** cohort — **Aravind Eye Hospital, Madurai**, 101 images with expert optic-disc/cup ground truth. That is the set we use when we say “Indian glaucoma fundus external.”

**Fourth — experiments I actually ran:**  
I took our AIROGS-trained fundus specialist (in-domain AUC about **0.97**) and tested it zero-shot on these externals.

- On **Drishti**, AUC lands around **0.89** overall and **0.95** on the official test split — so the Indian set **transfers**.  
- On **BEH** (Bangladesh), AUC about **0.78** — usable site-shift check.  
- On **ACRIMA** and **RIM-ONE**, zero-shot AUC collapses to about **0.57**. That is not a bug: those images are **optic-disc crops**; AIROGS is **full-field fundus**. Domain gap.

**Fifth — adaptation:**  
I fine-tuned the same model on ACRIMA. Test AUC went to **0.997**. So we **Adopt ACRIMA as adaptation data**, not as a zero-shot claim. **RIM-ONE** we still Adopt — but as a **hard multi-hospital stress test**, where we expect pain until we train multi-site.

**Recommendation in one line:**  
**Adopt Drishti, BEH, ACRIMA, and RIM-ONE for GLAUCO external / adaptation. Do not Adopt them into the DR product. BIOMISA can wait.**

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
2. `docs/external_DATASET_SPECS.md` (modality / India table)  
3. `docs/external_EXTERNAL_RESULTS.md` (full metrics)  
4. This brief: `docs/SUPERVISOR_external_BRIEF.md`
