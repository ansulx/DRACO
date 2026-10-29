# GHOTA — Novelty, modality conflict & produceable pipeline

**Audience:** mentor · AIIMS collaborators · internal team  
**Problem they raised:** mixing mismatched modalities → weak Acc, not production-ready, “nothing to show.”  
**Goal:** one clean story that is **novel**, **mathematical**, **reproducible**, and **deployable**.

---

## 1. The conflict (name it clearly)

| Bad pattern (what looks “multimodal” but fails) | Why Acc / product dies |
|-------------------------------------------------|------------------------|
| Dump OCT + fundus + OCTA into **one** backbone | Different physics → feature collision |
| Fake pairs / random pairing for fusion Acc | Inflated or meaningless numbers |
| One severity head for **DR + glaucoma** | Clinical label conflict |
| Claim “fusion SOTA” without paired eyes | Reviewers / AIIMS reject |

**Mentor is right:** modality soup ≠ product.

---

## 2. The resolution (what GHOTA *is*)

**GHOTA = one clinical product language, modality-routed specialists, optional paired fusion.**

```
                    ┌─ fundus photo ──► Specialist_F (RETFound-CFP) ─┐
  image + meta ─────┼─ OCT B-scan ────► Specialist_O (RETFound-OCT) ─┼─► GHOTA 3-class
                    └─ (if same eye) ──► Late / x-attn fusion ────────┘   {0,1,2}
```

| Rule | Production meaning |
|------|--------------------|
| **Route by modality** | Always works; no conflict at inference |
| **Fuse only on real pairs** | OEFI-style same-eye keys only |
| **DR-only labels** | Glaucoma stays in **GLAUCO** |
| **OCTA = separate specialist later** | Not forced into CFP/OCT weights |

This turns “modality conflict” into **modality discipline** — that *is* the novelty vs naive concat.

---

## 3. Novelty angles (publishable)

Pick **one primary claim** + support claims (don’t claim everything).

### Primary (recommended)
> **Task-conditioned multi-modal DR severity with modality routing and honest paired fusion** — unified 3-class DR language across fundus and OCT without label/modality collapse.

### Mathematical / methods (what to write formally)
1. **Unified label map** \( y \in \{0,1,2\} \) across ICDR / MMRDR / OEFI-DR (explicit mapping table in paper).  
2. **Task embedding** \( \mathbf{e}_t \) concat with backbone features → \( p(y \mid x, t) \) (OCT vs fundus task id).  
3. **Optimization recipe** (citeable): Focal loss + ViT layer-wise LR decay + cosine warmup (RETFound fine-tune).  
4. **Paired fusion** only when eye key matches:  
   \( p_{\text{fuse}} = \tfrac12(p_O + p_F) \) (late) or cross-attn \( f(T_O, T_F) \) (early) — report both; **no unpaired fusion Acc**.  
5. **External protocol**: holdout (MMRDR/APTOS) + **OEFI paired** + binary DME collapse metrics (AUC/Sens/Spec).

### Secondary novelty (if OCTA access arrives)
> Extend the *same* severity ontology to OCTA via a third specialist — **not** by mixing OCTA into the fundus encoder.

DcardNet [arXiv 2006.05480](https://arxiv.org/abs/2006.05480) = related work / multi-level DR design; **not** our training set.

---

## 4. What already proves “we utilize & produce”

| Component | Produceable artifact | Result (show collaborators) |
|-----------|----------------------|-----------------------------|
| OCT specialist | `ghota_oct_dme_retfound_sota` | Acc **0.901**, F1 **0.831** |
| Fundus specialist | `ghota_fundus_dr_retfound_sota` | Acc **0.922**, F1 **0.858** |
| OEFI OCT external | eval JSON | DME AUC **~0.99** |
| **Real paired fusion** | 409 OEFI pairs | DME AUC **0.979**; DR Acc **0.780** (fusion best) |
| Configs + catalogs + scripts | repo | Re-runnable pipeline |

Message to AIIMS: *We are not modality-confused — we are modality-routed, and we only fuse on true pairs.*

---

## 5. Pipeline to freeze (production + paper)

```
raw data → catalogs (remap labels) → train specialist(s) → holdout eval
        → OEFI external / paired fusion eval → figures + metrics JSON → demo API
```

**Freeze for v1 product**
1. Inference router: `modality ∈ {fundus, oct}` → correct checkpoint  
2. Optional: if both paths for same `patient_eye` → late fusion  
3. Output: class + softmax + (optional) DME binary score \(1-p_0\)  
4. Never output a fused score from unpaired inputs

**Configs that define v1**
- `configs/ghota_oct_dme_retfound_sota.yaml`
- `configs/ghota_fundus_dr_retfound_sota.yaml`
- Pair catalog: `data/processed/ghota/catalogs/ghota_oefi_paired.csv`

---

## 6. Next implementations (priority = novelty × produceability)

| Priority | Work | Why |
|---------:|------|-----|
| **P0** | Train **learnable late fusion / cross-attn on OEFI pairs** + locked test split | Turns eval stub into a **paper method**; fixes “we only averaged probs” |
| **P0** | Single **`ghota.infer` / demo** CLI or small API (route + optional fuse) | Collaborators can *run* something |
| **P1** | Paper tables: holdout + OEFI DME + OEFI paired DR; ablation (OCT / F / fuse) | AIIMS-ready results pack |
| **P1** | Formal label-map + pairing algorithm appendix in `docs/` | Mathematical clarity |
| **P2** | OCTA specialist **only after** ROAD/OCTA-500 access | New modality without conflict |
| **P3** | Joint multitask single-run training | Optional; routing specialists already publishable |

---

## 7. What *not* to do (protect Acc & reputation)

- Don’t merge GLAUCO + GHOTA heads for one “super Acc.”  
- Don’t train fusion on random OCT↔fundus pairs.  
- Don’t put OCTA into RETFound-CFP and call it fusion.  
- Don’t hide that for **binary DME**, OCT alone can beat naive fusion — report honestly (strengthens trust).

---

## 8. One conclusion (say this to mentor / AIIMS)

> Previous weakness was **modality conflict from forced mixing**.  
> GHOTA fixes that with **(1) unified DR severity labels, (2) modality-routed SOTA specialists, (3) fusion only on real same-eye pairs (OEFI)**.  
> That is both **more novel** than a single concat model and **more produceable** for clinic: one input modality → one specialist; two paired modalities → fusion.  
> Results are locked and re-runnable; next step is trained paired fusion + a deployable inference demo — not more unfocused datasets.

**Show docs:** `MENTOR_FOUR_DATASETS.md` · `GHOTA_RESULTS.md` · `TRAINED_FUSION.md`
