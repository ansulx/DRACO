# GLAUCO deck — full speaking script

**Deck:** `GLAUCO_External_Fundus_Datasets.pptx` (repo root + `docs/`)  
**Rebuild:** `python scripts/build_glauco_external_fundus_pptx.py`  
**How to use:** Read almost as written. Aim **6–8 minutes**. Pause on bold numbers.  
**After the talk:** open `docs/GLAUCO_EDGE_METRICS.md` if they want the full metric card.

---

## Slide 1 — Title (10 seconds)

Good afternoon. This briefing is **GLAUCO only — glaucoma screening**.

I’ll walk through: **disease → datasets → classes → problem → methodology → results and graphs** — then the peak `.pth` packs for edge handoff.

---

## Slide 2 — How to start / talk order (20 seconds)

The order is fixed so the story is clean:

1. Disease  
2. Datasets  
3. Classes  
4. Problem  
5. Methodology  
6. Results and graphs  

I won’t jump to numbers first — I’ll set train vs external first, then land on Sens / Spec / AUC.

---

## Slide 3 — Disease: glaucoma (45 seconds)

**This is glaucoma.** Optic-nerve damage — often silent until late. Screening needs **high sensitivity** without flooding clinics with false positives — that’s why we also report **specificity**.

Inputs we automate: **colour fundus** photos and **OCT / RNFLT** maps.

Product boundary — please keep this clear:

- **GLAUCO** = glaucoma only  
- **GHOTA** = diabetic retinopathy — separate product  

The Downloads / Kaggle sets in this review are **glaucoma**. They do **not** go into GHOTA. Indian DR stays on **IDRiD / APTOS** under GHOTA.

---

## Slide 4 — Datasets we used (1 minute)

Say this carefully — this is the Acc honesty slide.

**What trained the peak packs:**

- **Fundus peak** → **AIROGS-light**  
- **OCT peak** → **Harvard-GD** RNFLT  

**Also in the pipeline, but not the peak Acc claim:**

- **ACRIMA** (from Downloads) → fine-tune / OD-crop adaptation only  
- **Harvard-GDP / GF** → scale benches  
- **GLEAM** → 4-class staging / fusion research  

**External only — never used to train peak Acc:**

- **Drishti-GS1** (India, Aravind)  
- **BEH** (Bangladesh)  
- **RIM-ONE** (Spain, multi-hospital)  
- **BIOMISA** cataloged; later — not in peak  

One line: *Peak fundus Acc is AIROGS. Peak OCT Acc is Harvard-GD. Externals prove transfer.*

---

## Slide 5 — Classes (30 seconds)

**Peak packs are 2-class:** `normal` vs `glaucoma`.

That’s what sits in:

- `glauco_fundus_peak.pth`  
- `glauco_oct_rnflt_edge.pth`  

Separately we have a **4-class** staging track on GLEAM (`normal → early → mid → advanced`). Three-class is possible if required. For edge handoff we froze **binary screening**.

---

## Slide 6 — Problem (40 seconds)

Four problems we set out to solve:

1. **Screening** — detect glaucoma from fundus and/or RNFLT with measurable Sens / Spec.  
2. **Train vs test honesty** — primary Acc on AIROGS + Harvard-GD; externals never touch peak Acc.  
3. **Domain shift** — OD-crop sets (ACRIMA / RIM-ONE) break full-field models; that needs adaptation, not a fake zero-shot claim.  
4. **Edge handoff** — ship `.pth` plus architecture, params, memory, and a full metric card for the deploy team.

---

## Slide 7 — Methodology (45 seconds)

Six steps:

**A.** Catalog Downloads / Kaggle into GLAUCO CSVs.  
**B.** Train fundus specialist — RETFound-CFP on AIROGS-light.  
**C.** Train OCT specialist — EfficientNet-B0 on Harvard-GD.  
**D.** Adapt — fine-tune ACRIMA for OD-crop domain (separate checkpoint).  
**E.** External eval — zero-shot on Drishti / BEH / ACRIMA / RIM-ONE; report Sens / Spec / AUC.  
**F.** Export — freeze `glauco_edge_v1` `.pth` + model card.

---

## Slide 8 — Results: peak metrics (1 minute)  ← *Sens / Spec block*

This is the metric checklist slide.

**Fundus peak — AIROGS holdout:**

| Metric | Value |
|--------|------:|
| Accuracy | **0.920** |
| Precision | **0.920** |
| Recall / Sensitivity | **0.920** |
| Specificity | **0.920** |
| F1 | **0.920** |
| AUC | **0.969** |

Extra clinical knob: **Sens @ 95% Spec ≈ 0.83**.

**OCT peak — Harvard-GD test (edge-friendly, improved):**

| Metric | Value |
|--------|------:|
| Accuracy | **0.870** |
| Precision | **0.865** |
| Recall / Sensitivity | **0.882** |
| Specificity | **0.857** |
| F1 | **0.874** |
| AUC | **0.906** |

How to say it: fundus is the heavy Acc peak. OCT is **edge-friendly** — **4M params**, **~15 MB**, Acc **0.87** / Sens **0.88** after Acc-monitor retrain + thr **0.55** (was Acc 0.83 / Sens 0.77).

---

## Slide 9 — Graph: external fundus AUC (1 minute)  ← *graphs start here*

Same peak fundus `.pth`, **zero-shot** — no retraining.

Point at the bars left to right:

- **AIROGS holdout ~0.97** — in-domain  
- **Drishti India 0.89** / official test **0.95** — **Indian transfer works**  
- **BEH ~0.78** — usable South-Asia site shift  
- **ACRIMA / RIM-ONE ~0.57** — expected fail: those are **optic-disc crops**; AIROGS is **full-field**  

Fix for OD-crop: fine-tune on ACRIMA → test AUC **0.997**.  
RIM-ONE stays the **hard multi-hospital stress test**.

---

## Slide 10 — Graph: fundus vs OCT six metrics (40 seconds)

Side-by-side bars: Acc, Prec, Sens, Spec, F1, AUC.

One sentence: *Fundus peaks accuracy; OCT peaks deployability. Both binary. Both shipped as packs.*

---

## Slide 11 — External Sens / Spec table (45 seconds)

Walk the India row first:

- **Drishti full:** AUC **0.890** · Acc **0.812** · Sens **0.786** · Spec **0.871** → adopt as Indian external  
- **BEH:** AUC **0.782** · Sens lower (**0.585**) · Spec **0.870** → light adopt as site-shift  
- **ACRIMA / RIM-ONE:** Sens collapses zero-shot → hire for adaptation / hard bench, not as a zero-shot Acc claim  

After ACRIMA FT: ACRIMA test **AUC 0.997 · Acc 0.972 · Sens@95%Spec 0.987**. Drishti/BEH still hold; RIM-ONE hospital OOD still hard.

---

## Slide 12 — Peak packs for edge (30 seconds)

What we hand the deploy team:

| Pack | Train | Arch | Size | Headline |
|------|-------|------|------|----------|
| `glauco_fundus_peak.pth` | AIROGS-light | RETFound-CFP · 303M | ~1.1 GB | Acc 0.92 · AUC 0.97 |
| `glauco_oct_rnflt_edge.pth` | Harvard-GD | EfficientNet-B0 · 4M | ~15.5 MB | Acc 0.87 · Sens 0.88 · AUC 0.91 |

Each pack includes the full metric card: Sens / Spec / Prec / Rec / F1 / Acc / AUC + params / memory.

---

## Slide 13 — Speak this (optional crib) (15 seconds)

Only if you lose the thread — six lines:

1. These are glaucoma datasets under GLAUCO — not DR.  
2. Train: AIROGS + Harvard-GD. Externals: Drishti, BEH, ACRIMA, RIM-ONE.  
3. Peak packs = 2-class.  
4. Binary screening with honest train/test.  
5. Specialists → OOD → adapt OD-crop → export.  
6. Fundus AUC 0.97 · OCT AUC 0.91 · Drishti 0.89–0.95 — then show the bars.

---

## Slide 14 — Ask / close (20 seconds)

**Decision we need:**

Confirm peak packs stay binary:

- fundus = **AIROGS-light**  
- OCT = **Harvard-GD**  
- externals stay **test-only**  

Detail docs if they want more: `docs/GLAUCO_DATASETS_USED.md` · `docs/GLAUCO_EDGE_METRICS.md` · `exports/glauco_edge/`.

---

## Four numbers to memorize (backup)

| Story | Number |
|-------|--------|
| Fundus in-domain AUC | **0.97** |
| Indian Drishti external | **0.89** full / **0.95** official |
| OCT Harvard-GD Acc / Sens / AUC | **0.87 / 0.88 / 0.91** |
| ACRIMA after FT | **0.997** (zero-shot was ~0.57) |

---

## If they interrupt

| Question | Answer |
|----------|--------|
| Why not train peak Acc on Drishti? | n=101 — classic Indian **external** bench, not primary train scale. |
| Why is ACRIMA zero-shot bad? | OD-crop vs full-field — domain gap, not a label bug. |
| Why binary not 4-class for edge? | Screening Acc for deploy; staging stays on GLEAM research track. |
| Why not put these in GHOTA? | Wrong disease / wrong labels. DR stays APTOS/IDRiD. |
| Where are the `.pth` files? | Local `exports/glauco_edge/` — large weights gitignored; cards + docs are in repo. |
