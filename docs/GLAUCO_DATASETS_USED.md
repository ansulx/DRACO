# GLAUCO — exact datasets used (defend this)

**Product:** glaucoma only.  
**Peak edge pack shipped:** **2 classes** — `0 = normal`, `1 = glaucoma`.  
**Why not 3 by default:** screening Acc + edge handoff is binary; we *also* trained graded models (see below) when the dataset supports it.

---

## 1. Classes — what to say

| Setting | Classes | Labels | Where |
|---------|--------:|--------|--------|
| **Peak fundus pack (edge)** | **2** | normal / glaucoma | `glauco_fundus_peak.pth` |
| **Peak OCT RNFLT (edge-friendly)** | **2** | normal / glaucoma | `glauco_oct_rnflt_edge.pth` |
| GLEAM staging models | **4** | normal → early → intermediate → advanced | trained, research track |
| GAMMA (config exists) | **3** | normal / early / advanced | fusion config; use if they insist on 3-class |

**Defense line:**  
> “Deploy pack is **binary screening** (referable glaucoma vs normal) — that’s the clinical Acc target for edge. We also trained **4-class staging on GLEAM**. Three-class is possible via GAMMA-style grades; binary is what we froze for the `.pth` handoff.”

---

## 2. Datasets we **TRAINED** on (weights learned here)

| # | Dataset | Modality | Classes | Role | On disk |
|---|---------|----------|--------:|------|---------|
| 1 | **AIROGS-light** | Colour fundus | 2 | **Primary fundus Acc** (RETFound-CFP) | was used; folder may be off machine now |
| 2 | **Harvard-GD** | RNFLT | 2 | OCT/RNFLT baseline + edge OCT pack | ✓ `data/raw/glaucoma/harvard_gd` |
| 3 | **Harvard-GDP** | RNFLT | 2 | OCT scale | ✓ |
| 4 | **Harvard-GF** | RNFLT | 2 | Large OCT benchmark | ✓ |
| 5 | **GLEAM** | Fundus + OCT + VF | 2 or **4** | Fusion / staging | ✓ `GLEAM/` |
| 6 | **ACRIMA** | Fundus (OD-crop) | 2 | Fine-tune adaptation (from Downloads) | ✓ `data/raw/glaucoma/acrima` |

**Primary peak fundus model = trained on AIROGS-light only** (then ACRIMA FT is a separate adapted checkpoint).

---

## 3. Datasets from **Downloads** (your ask)

These came from Windows Downloads / Kaggle into `data/raw/glaucoma/`:

| Dataset | From Downloads? | Used how? |
|---------|-----------------|-----------|
| **ACRIMA** | ✓ `archive.zip` | **Trained** (fine-tune) + external eval |
| **BEH** | ✓ BEH zip | **External eval only** (not train) |
| **BIOMISA** | ✓ `Dataset.zip` | Cataloged; **not** in peak Acc train |
| **Drishti-GS1** | ✓ (Kaggle mirror; not IIIT form) | **External eval only** (Indian set) |
| **RIM-ONE DL** | Kaggle (not in Downloads folder) | **External eval only** |

**Defense line:**  
> “From Downloads we brought in ACRIMA, BEH, BIOMISA; plus Drishti and RIM-ONE. **Only ACRIMA entered training** (adaptation). The others are **held-out external tests** so we don’t leak Acc.”

---

## 4. What the **peak `.pth`** was built from

| Item | Value |
|------|--------|
| Checkpoint | `checkpoints/glauco_retfound_cfp_airogs_light/best.pt` |
| **Train dataset** | **AIROGS-light** (fundus, binary) |
| Architecture | RETFound-CFP |
| Classes | **2** — normal / glaucoma |
| Holdout Acc / AUC | ~0.92 / ~0.97 |
| Externals tested | Drishti, BEH, ACRIMA, RIM-ONE |

Companion edge OCT `.pth` → **Harvard-GD** only (EfficientNet-B0, 2-class).

---

## 5. One-screen map (train vs test)

```
TRAIN (learns weights)
  AIROGS-light ──────────► peak fundus .pth          [2-class]
  Harvard-GD/GDP/GF ─────► OCT/RNFLT models          [2-class]
  GLEAM ─────────────────► fusion / 4-stage models   [2 or 4]
  ACRIMA (FT) ───────────► adapted OD-crop model     [2-class]

EXTERNAL ONLY (never train peak on these)
  Drishti (India) · BEH · RIM-ONE · ACRIMA full zero-shot
  BIOMISA (kept for later; small OD n)
```

---

## 6. Short defend script (30–45 s)

> For glaucoma we ship a **2-class** screening model: normal vs glaucoma.  
> The peak fundus weights were trained on **AIROGS-light**.  
> From Downloads we added **ACRIMA, BEH, BIOMISA**, plus **Drishti** and **RIM-ONE** — Drishti is our **Indian** external; ACRIMA we also **fine-tuned** on.  
> BEH / Drishti / RIM-ONE stay **test-only** so Acc is honest.  
> We also have **4-class GLEAM** staging if they want severity; the edge `.pth` is binary because that’s the deploy Acc target.

---

## 7. OCT / RNFLT — datasets used

### TRAINED (OCT-side)

| # | Dataset | What it is | Classes | Best Acc signal | Edge pack? |
|---|---------|------------|--------:|-----------------|------------|
| 1 | **Harvard-GD** | RNFLT maps (smoke → solid Acc) | **2** | AUC **0.913** · Acc **0.83** (EfficientNet) | **Yes** → `glauco_oct_rnflt_edge.pth` |
| 2 | **Harvard-GDP** | RNFLT, larger / progression | **2** | AUC **0.855** | No (research) |
| 3 | **Harvard-GF** | RNFLT (+ B-scan available) | **2** | AUC **0.838** | No (research) |
| 4 | **GLEAM OCT** | cpOCT thickness (paired) | **4** staging | Specialist Acc ~0.52 (hard) | Via fusion track |
| 5 | **GLEAM fusion** | OCT + fundus (+ VF) | **2** or **4** | Trimodal binary AUC **0.937** | Fusion, not single-OCT edge |

Also: **RETFound-RNFLT** on Harvard-GD (AUC **0.883**) — heavy; EfficientNet GD is the **edge OCT** choice.

### OCT from Downloads?

| Dataset | OCT? | Used for OCT train? |
|---------|------|---------------------|
| **BIOMISA** | Yes (volumes + B-scans + fundus) | **Not yet** in peak OCT train — cataloged only |
| ACRIMA / BEH / Drishti / RIM-ONE | Fundus only | N/A for OCT |

**Defense line (OCT):**  
> “OCT/RNFLT peak for edge is **Harvard-GD** EfficientNet — 2-class, ~4M params, ~15 MB, AUC 0.91. We also scaled on Harvard-GDP/GF and fused OCT with fundus on GLEAM. Downloads BIOMISA has OCT but wasn’t used for the shipped OCT `.pth`.”

### Peak OCT `.pth`

| Item | Value |
|------|--------|
| File | `exports/glauco_edge/glauco_oct_rnflt_edge.pth` |
| **Train dataset** | **Harvard-GD** |
| Architecture | EfficientNet-B0 |
| Classes | **2** — normal / glaucoma |
| Params / weights | **4.01 M** / **15.5 MB** |
| AUC / Acc | **0.913** / **0.83** |

---

## 8. Quick answers

| Question | Answer |
|----------|--------|
| Can it be 3 classes? | Yes in principle (GAMMA-style). Peak pack is **2** for screening. GLEAM gives **4**. |
| Which Downloads sets (fundus)? | ACRIMA, BEH, BIOMISA (+ Drishti/RIM-ONE via Kaggle). |
| Which one trained the peak fundus? | **AIROGS-light**. |
| Which one trained the peak OCT? | **Harvard-GD**. |
| Did Downloads train OCT? | **No** — BIOMISA OCT is on disk, not in the edge OCT `.pth`. |
| Did you train on Drishti? | **No** — external only (defend generalization). |
| Best Indian story? | **Drishti-GS1** (Aravind) external AUC ~0.89–0.95. |