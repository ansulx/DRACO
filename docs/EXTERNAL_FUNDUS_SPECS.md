# External fundus candidates — disease, modality, specs (supervisor)

**All five Oct-16 fundus links are GLAUCOMA (or glaucoma + other eye disease), not DR Acc packs.**  
They belong under **GLAUCO**, not GHOTA. GHOTA already has separate DR fundus (APTOS / IDRiD).

---

## One-screen matrix (on disk ✓)

| Dataset | Disease | Modality | Task | n (on disk) | Site / region | **India?** | Role for us |
|---------|---------|----------|------|-------------|---------------|------------|-------------|
| **Drishti-GS1** | **Glaucoma** | Colour fundus (OD-centered) + OD/cup GT | Binary G vs N + ONH seg / CDR | **101** (70G / 31N) · 50/51 split | **Aravind Eye Hospital, Madurai** | **YES** | Indian external / CDR story |
| **ACRIMA** | **Glaucoma** | Colour fundus (OD-cropped) | Binary G vs N | **705** (396G / 309N) | FISABIO, Valencia, **Spain** | No | Largest easy Acc external |
| **RIM-ONE DL** | **Glaucoma** | Colour fundus (OD-cropped) + OD/cup masks | Binary G vs N | **485** (172G / 313N) | 3 hospitals, **Spain** | No | Hospital-split domain shift |
| **BEH** | **Glaucoma** | Colour fundus (full / OD) | Binary G vs N | **634** (171G / 463N) | Bangladesh Eye Hospital, Dhaka | No (BD) | South-Asia site shift |
| **BIOMISA** (Mendeley) | **Glaucoma** + macular (DME/CSR/AMD) | **Fundus + OCT** (composite) | Multi-disease; OD G vs Healthy | OD Color **79** frames / **44** pts | **NUST / Pakistan** (+ KU) | No (PK) | Small paired sandbox; not DR Acc |

---

## Specs worth saying out loud

### Drishti-GS1 — **the Indian set in this batch**
- **101** PNG fundus, FOV 30°, ~2896×1944, OD-centered  
- 4 expert OD/cup markings · soft maps · CDR · image-level G/N  
- Train GT public; classic Indian ONH/glaucoma benchmark  

### ACRIMA (Spain)
- 705 OD-cropped JPG · label in filename (`_g_` = glaucoma)  
- Expert annotated · strong plug-and-play Acc external  

### RIM-ONE DL (Spain)
- 485 PNG · official **by-hospital** train/test (domain shift)  
- Also random partition · OD/cup reference segs available  

### BEH (Bangladesh)
- 634 JPG · Topcon TRC-50DX · ages ~35–80 · 2 ophthalmologists  
- Imbalanced (463 normal / 171 G) — good stress test  

### BIOMISA composite (Pakistan)
- **Dual modality:** Colour/red-free fundus + OCT B-scans / volumes  
- Branches: **OD** (Glaucoma / Healthy) · **Macula** (Healthy, ci-DME, CSR, AMD)  
- Macula DME ≠ APTOS/IDRiD ICDR grades → do **not** fold into GHOTA Acc  

---

## India / South Asia framing for supervisor

| Bucket | Datasets | Use in talk |
|--------|----------|-------------|
| **Indian (this external batch)** | **Drishti-GS1 only** | “Indian glaucoma fundus + ONH GT from Aravind” |
| **Other South Asia** | BEH (BD), BIOMISA (PK) | Regional external / site shift |
| **EU glaucoma fundus** | ACRIMA, RIM-ONE DL | Size + hospital-shift benchmarks |
| **Indian DR (already GHOTA)** | **IDRiD**, **APTOS** (Aravind-linked fundus DR) | Keep on diabetic track — not these five |

**Line for PPT:**  
> external’s five recent links are **glaucoma fundus** (one composite fundus+OCT). Only **Drishti-GS1** is an **Indian** cohort; we also hold **IDRiD/APTOS** as Indian **DR** fundus for GHOTA.

---

## Not in this five (earlier external DR/OCTA refs)

| Resource | Disease | Modality | India? | Status |
|----------|---------|----------|--------|--------|
| OEFI | DR/DME | Fundus + OCT paired | No | On disk · fusion done |
| OCTA-500 | Multi / vessel | OCT + **OCTA** | No | Access pending |
| ROAD | DR grades | **WF-OCTA** | No | Access pending |
| DcardNet | DR | OCT+OCTA (private) | No | Cite only |

---

## GHOTA vs GLAUCO routing

```
These five fundus sets  ──►  GLAUCO (glaucoma)     ✓ downloaded + catalogs
IDRiD / APTOS      ──►  GHOTA  (DR Acc)       ✓ fundus DR path
OEFI               ──►  GHOTA  fusion         ✓
OCTA-500 / ROAD    ──►  future OCTA branch    ✗ not yet
```

## Experiments (started)

See **[external_EXTERNAL_RESULTS.md](external_EXTERNAL_RESULTS.md)**.

| Adopt | How we use it | Headline |
|------|---------------|----------|
| Drishti | Zero-shot external | AUC **0.89** full / **0.95** official test |
| BEH | Zero-shot external | AUC **0.78** |
| ACRIMA | Fine-tune + test | After FT: AUC **0.997** (zero-shot was 0.58) |
| RIM-ONE | Hard OOD bench | Zero-shot AUC ~0.57 — keep as stress test |
