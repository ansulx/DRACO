# Supervisor meeting — full speaking script

**How to use:** Read this almost as-is. Aim **8–12 minutes**. Bold lines = say slowly.  
**Deck:** `SUPERVISOR_Showcase.pptx`  
**Handout:** `docs/SUPERVISOR_SHOWCASE.md`

---

## 0. Opening (30 seconds)

Good afternoon. I’ll walk through what we’ve built, in plain order:

1. **What problem** we are solving  
2. **What datasets** we used  
3. **What modalities**  
4. **What technique / workflow**  
5. **What results** we have now  
6. **What problems remain** and what we plan next  

The important design choice up front: we do **not** put everything into one confused model. We have **two products**:

- **GLAUCO** → glaucoma  
- **GHOTA** → diabetic retinopathy / DME, as a **3-class** severity score  

That split is deliberate — it is how we avoid the modality and label conflicts we were hitting before.

---

## 1. The problem (1 minute)

Clinically we care about two different diseases:

- **Glaucoma** — optic nerve / RNFL / visual field  
- **Diabetic retinopathy and macular edema** — retina damage from diabetes  

If we train one head that mixes “advanced glaucoma” with “advanced DR,” the word *advanced* means different things. Accuracy drops, and a clinic cannot trust the output.

Separately, images come from **different cameras**:

- Colour **fundus** photo  
- Structural **OCT** B-scan  
- Sometimes **OCTA** (blood-flow angiography) — that is *another* physics  

If we dump all of that into one backbone without discipline, we get what my mentor called **modality conflict**: looks multimodal on a slide, but is weak in production.

So our rule is:

> **One disease family per product. Route by modality. Fuse only on real same-eye pairs.**

---

## 2. Modalities — what goes where (1 minute)

Please think of three boxes:

| Modality | What it is | Used in |
|----------|------------|---------|
| **Fundus** | Colour photo of the retina | GLAUCO + GHOTA |
| **OCT B-scan** | Structural cross-section | GLAUCO (RNFLT/OCT) + GHOTA |
| **OCTA** | Vessel / flow map | *Not in production yet* — next phase |

For **GHOTA** (diabetes) today:

- Fundus photo → fundus specialist  
- OCT B-scan → OCT specialist  
- If we have **both from the same eye** → fusion  

For **GLAUCO** (glaucoma):

- Fundus screening  
- RNFLT / OCT  
- Optional GLEAM fusion (fundus + OCT + visual field)

---

## 3. Datasets — “this was the dataset” (2 minutes)

### Glaucoma — GLAUCO

| Dataset | Role |
|---------|------|
| **AIROGS-light** | Fundus glaucoma / not-glaucoma |
| **Harvard-GD / GDP / GF** | RNFLT / OCT glaucoma benchmarks |
| **GLEAM** | Multi-modal fusion set (fundus + OCT + VF) |

### Diabetic — GHOTA (3-class)

| Dataset | Role |
|---------|------|
| **MMRDR** | Main OCT DME 3-class train/test |
| **OCTID** | Extra OCT (normal / DR) |
| **APTOS 2019** | Main fundus DR grades → mapped to 3-class |
| **IDRiD** | Extra fundus grading (same modality) |
| **OEFI** | External OCT + **paired** fundus+OCT (same patient/eye names) |
| **Kermany** | Large OCT corpus (optional scale; different label scheme) |

### The four links you asked us to study

| Link | What it is | Did we use it? |
|------|------------|----------------|
| **OEFI** (GitHub OCT + fundus) | Same modalities as GHOTA, **paired** | **Yes — fully used** |
| **OCTA-500** | OCT + OCTA volumes | Not yet — needs IEEE access |
| **ROAD** | OCTA DR grades | Not yet — needs author email |
| **DcardNet paper** | Private OHSU OCT+OCTA method | Cite only — no public data |

**Sentence for supervisor:**  
“Among those four, only OEFI matches our current production modalities. We downloaded it, built same-eye pairs, and trained fusion on it. OCTA-500 and ROAD are the next modality branch, not something we force into today’s fundus/OCT models.”

---

## 4. Technique — “this is how we train” (1.5 minutes)

### Backbone
We use **RETFound** (Nature foundation model):

- RETFound-OCT for OCT  
- RETFound-CFP for colour fundus  

### Training recipe (SOTA fine-tune style)
- **Focal loss** (handles rare classes)  
- **Layer-wise learning-rate decay** on the ViT  
- **Cosine schedule + warmup**  
- Short **head-only warmup**, then unfreeze backbone  

### GHOTA label language (3-class)
We map every DR/DME source into one clinical language:

| Class | Meaning |
|------:|---------|
| **0** | None |
| **1** | Early–moderate (mild/moderate grades) |
| **2** | Advanced (severe / PDR / center-involving DME) |

So OCT DME grades and fundus ICDR grades speak the **same severity language**, but through **separate specialists**.

### Fusion
- Specialists stay frozen  
- On **real OEFI pairs**, we train fusion (best = **cross-attention**)  
- At inference: if only one image → specialist; if paired OCT+fundus → fusion  

CLI:

```text
python -m ghota.infer --image X --modality oct|fundus
python -m ghota.infer --oct O.jpg --fundus F.jpg --fusion-ckpt ...
```

---

## 5. Workflow — “this is the pipeline I’m thinking” (1.5 minutes)

Walk this as a flowchart out loud:

```
Raw images
   ↓
Catalogs (paths + remapped labels + modality + task id)
   ↓
Train specialists (GLAUCO glaucoma / GHOTA DR)
   ↓
Holdout evaluation
   ↓
External evaluation (e.g. OEFI)
   ↓
If same-eye pair available → fused prediction
   ↓
Demo / clinic-facing output: class + confidence
```

**Production behavior**

1. User uploads **one** image → we detect or ask modality → correct specialist → 3-class (GHOTA) or glaucoma score (GLAUCO).  
2. User uploads **paired** OCT + fundus of the **same eye** → fusion model.  
3. We **never** invent pairs. We **never** mix glaucoma labels into GHOTA.

That is the workflow I want locked for the thesis / paper / AIIMS demo.

---

## 6. Results — say the numbers slowly (2 minutes)

### GLAUCO
- Fundus screening: **AUC 0.983**  
- RNFLT (Harvard-GD): **AUC 0.913**  
- GLEAM fusion binary: AUC about **0.95**; trimodal Acc about **0.76**  

### GHOTA specialists
- OCT holdout: **Acc 0.901**, Macro-F1 **0.831**  
- Fundus holdout: **Acc 0.922**, Macro-F1 **0.858**  

### External OEFI (OCT → DME yes/no)
- **AUC 0.994**, Sens **0.952**, Spec **0.969**  

### Trained paired fusion (patient holdout)
- Cross-attention: **Acc 0.802**, Macro-F1 **0.714**  
- Beats OCT-only / fundus-only / simple average on that split  

**One proud sentence:**  
“We have strong single-modality Acc, strong external DME AUC, and an honest fused Acc on real pairs — not on fake pairing.”

---

## 7. Problems we still see / will see later (1.5 minutes)

Be honest — supervisors trust this:

1. **OCTA not integrated yet**  
   OCTA-500 and ROAD need access. Until then we don’t claim angiography Acc.

2. **Class imbalance**  
   Advanced class and rare NCI-DME are harder; Macro-F1 still trails Acc on some splits.

3. **Paired data is still limited**  
   OEFI gives 409 pairs — enough to prove the method, not enough to claim “solved fusion forever.”

4. **Kermany labels ≠ GHOTA 3-class**  
   Useful for scale / pretraining, dangerous if merged blindly.

5. **GLEAM multi-class Acc vs HAMM**  
   Our binary AUC is strong; multi-class Acc still has headroom vs the 2026 HAMM bar (~0.81 Acc).

6. **Clinic deployment**  
   We have CLI inference; a polished UI / hospital integration is future work.

---

## 8. What I’m asking from you (supervisor) (45 seconds)

1. Please **confirm** the GLAUCO ⊥ GHOTA split — I think this is the right scientific and product story.  
2. Please **approve** starting the OCTA access process (OCTA-500 + ROAD) as Phase 2.  
3. For publication, I’d like the novelty framed as:  
   **task-conditioned dual specialists + unified 3-class DR language + honest paired fusion.**  
4. If useful, I can live-demo: one OCT, one fundus, and one paired fusion call.

---

## 9. Closing (20 seconds)

To summarize:

- **Datasets:** Harvard / AIROGS / GLEAM for glaucoma; MMRDR / APTOS / OEFI / IDRiD for diabetes.  
- **Techniques:** RETFound + focal + layer-decay; modality routing; cross-attn fusion on real pairs.  
- **Workflow:** catalog → specialist → optional fuse → class + confidence.  
- **Results:** glaucoma AUC up to ~0.98; DR Acc ~0.90–0.92; OEFI AUC ~0.99; fused Acc ~0.80.  
- **Next:** OCTA when access lands; more pairs; deployment UI.  

Happy to take questions.

---

## Quick Q&A cheat sheet

**Q: Why two models not one?**  
A: Different diseases and different cameras. Mixing hurt Acc. Routing is more accurate and more deployable.

**Q: Is fusion SOTA?**  
A: For *our* paired OEFI holdout, trained cross-attn beats specialists and average fusion. We don’t claim a universal SOTA without more public paired sets.

**Q: Why ignore OCTA-500 / ROAD for now?**  
A: Different modality (angiography). Involving them means a new specialist, not stuffing them into fundus/OCT weights.

**Q: Can AIIMS run this?**  
A: Yes at CLI level today (`ghota.infer` / glauco infer). UI packaging is next if they want a pilot.

**Q: What’s novel?**  
A: Unified DR 3-class ontology across OCT and fundus, modality-conditioned specialists, and fusion trained only on true same-eye pairs — addressing modality conflict explicitly.
