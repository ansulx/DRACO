# Supervisor showcase — GLAUCO + GHOTA + reference datasets

**One line:** We split diseases cleanly — **GLAUCO = glaucoma**, **GHOTA = diabetic 3-class severity** — with modality routing (no fake mixing) and real paired fusion on OEFI.

**Open these first**
1. `SUPERVISOR_Showcase.pptx` (this folder / repo root)
2. This file
3. Optional deep dives: `GLAUCOMA_RESULTS_2026.md` · `GHOTA_RESULTS.md` · `MENTOR_FOUR_DATASETS.md`

---

## 1. Architecture (why Acc improved)

| Track | Disease | Modalities | Output |
|-------|---------|------------|--------|
| **GLAUCO** | Glaucoma | Fundus · RNFLT/OCT · VF (GLEAM) | Binary / staging |
| **GHOTA** | DR / DME only | Fundus · OCT B-scan · paired fusion | **3-class:** none / early–mod / advanced |

**Modality discipline:** route by input type → specialist. Fuse **only** on real same-eye pairs. Do not dump OCTA into fundus/OCT weights.

---

## 2. GLAUCO — glaucoma updates (show numbers)

| Model | Result |
|-------|--------|
| Fundus RETFound (AIROGS-light) | AUC **0.983**, F1 **0.936** |
| OCT/RNFLT Harvard-GD | AUC **0.913**, F1 **0.830** |
| OCT/RNFLT Harvard-GF | AUC **0.838**, F1 **0.772** |
| GLEAM trimodal late fusion | Acc **0.762**, F1 **0.704**, binary AUC **0.947** |
| GLEAM trimodal binary | Acc **0.867**, AUC **0.955** |

Lit bar (HAMM tri-modal): Acc ~0.81 · F1 ~0.76 · AUROC ~0.93 — we are competitive on binary AUC; multi-class Acc still the stretch goal.

**Demo:** `docs/glauco_ar_demo/` (3D/AR) · `docs/GLAUCO_Presentation_3Slides.pptx`

---

## 3. GHOTA — 3-class DR updates

| Specialist | Holdout Acc | Macro-F1 |
|------------|------------:|---------:|
| OCT (MMRDR + OCTID) | **0.901** | **0.831** |
| Fundus (APTOS) | **0.922** | **0.858** |

| External / fusion | Result |
|-------------------|--------|
| OEFI OCT → binary DME | AUC **0.994** · Sens **0.952** · Spec **0.969** |
| Trained cross-attn fusion (patient holdout) | Acc **0.802** · F1 **0.714** |

**Infer demo**
```powershell
.\.venv\Scripts\python.exe -m ghota.infer --image <path> --modality oct
.\.venv\Scripts\python.exe -m ghota.infer --oct <o.jpg> --fundus <f.jpg> --fusion-ckpt checkpoints\ghota_oefi_fusion_xattn\best.pt
```

---

## 4. Four links (Parth / mentor) — status update

| # | Link | Same modality? | What we did |
|---|------|----------------|-------------|
| 1 | [OEFI OCT+Fundus](https://github.com/Traslational-Visual-Health-Laboratory/OCT-AND-EYE-FUNDUS-DATASET) | **Yes** | Full download · **409 pairs** · fusion trained |
| 2 | [OCTA-500](https://ieee-dataport.org/open-access/octa-500) | Partial (OCTA new) | Access gated (IEEE + email) — next after approval |
| 3 | [ROAD](https://mip2019.github.io/ROAD/) | Partial (OCTA) | Author email needed — maps to GHOTA 0/1/2 |
| 4 | [DcardNet 2006.05480](https://arxiv.org/abs/2006.05480) | Idea only | Cite as related work — **no public data** |

**Same-modality extras also on disk:** IDRiD fundus · Kermany OCT (optional scale).

---

## 5. Talking points (30–60 s)

1. “We fixed modality conflict by **splitting products**: GLAUCO vs GHOTA.”  
2. “Glaucoma fundus AUC **0.98**; DR specialists Acc **0.90 / 0.92**.”  
3. “Only OEFI among the four links matches our modalities — we **used it** for real paired fusion Acc **0.80**.”  
4. “OCTA-500 / ROAD are the next modality branch, not forced into today’s backbone.”  

---

## 6. File checklist for the meeting

| File | Use |
|------|-----|
| `SUPERVISOR_Showcase.pptx` | Main deck |
| `docs/SUPERVISOR_SHOWCASE.md` | Handout / leave-behind |
| `GHOTA_Presentation_3Slides.pptx` | DR deep dive |
| `docs/GLAUCO_Presentation_3Slides.pptx` | Glaucoma deep dive |
| `docs/ghota/figures/*.png` | Acc / OEFI / modality matrix |
| `results/glaucoma_figures/*.png` | GLEAM vs HAMM |
