# Supervisor comparison — DRACO OCT vs MMRDR 2026 paper

Reference: *Scientific Data* 2026, Supplementary **Table S2.4** (MMRDR-OCT, 3-class DME).

## Medical terms (quick)

| Term | Meaning |
|------|---------|
| **Precision** | Of predicted positives, how many are correct |
| **Recall (Sensitivity)** | Of actual positives, how many we catch |
| **Specificity** | Of actual negatives, how many we correctly rule out |
| **F1** | Balance of precision & recall |
| **Sens @ 95% Spec** | Sensitivity when specificity is fixed at 95% (screening trade-off) |

## MMRDR test (562) — vs paper RETFound

| Metric | Paper RETFound | DRACO RETFound v2 | Δ |
|--------|----------------|-----------------|---|
| Accuracy | 0.897 | 0.890 | -0.007 |
| Macro-F1 | 0.759 | 0.784 | +0.025 |
| F1 No DME | 0.923 | 0.929 | +0.006 |
| F1 NCI-DME | 0.436 | 0.506 | +0.070 |
| F1 CI-DME | 0.919 | 0.916 | -0.003 |

**Headline:** We **beat paper macro-F1 (0.784 vs 0.759)** and **NCI-DME F1 (+0.07)** — the hardest rare class.

## OEFI external (1,113) — binary DME

| Metric | RETFound v2 |
|--------|-------------|
| AUC | 0.994 |
| Sensitivity @ 0.5 | 0.928 |
| Specificity @ 0.5 | 0.978 |
| Sens @ 95% Spec | 0.970 |

## Do we need more training?

| Question | Answer |
|----------|--------|
| More OCT DME on MMRDR? | **Optional, not critical.** Best model already beats paper F1. Gains would be marginal on accuracy; NCI class could still improve. |
| More DR (5-class) training? | **Different pipeline (fundus).** OCT task is DME staging, not ICDR 0–4. |
| Higher model? | RETFound ViT-L is already paper SOTA backbone. Focal loss did not beat weighted CE. |

## Figures

- `figures/01_macro_f1_vs_paper.png`
- `figures/02_per_class_f1_retfound_vs_paper.png`
- `figures/03_precision_recall_f1_ours.png`
- `figures/04_oefi_sensitivity_specificity.png`
- `figures/05_delta_vs_paper_retfound.png`
- `DRACO_supervisor_slide.png` — **single combined slide (16:9)**
