# GHOTA — DR-only 3-class severity (OCT + fundus)

**Scope:** Diabetic retinopathy / DME only.  
**Not in scope:** Glaucoma → stays in **GLAUCO**.  
**Priority:** Accuracy first.  
**Results pack:** [`GHOTA_RESULTS.md`](GHOTA_RESULTS.md) · slides: `GHOTA_Presentation_3Slides.pptx`

---

## Latest scores (SOTA specialists)

| Specialist | Data | Acc | Macro-F1 |
|------------|------|----:|---------:|
| **OCT** | MMRDR + OCTID | **0.901** | **0.831** |
| **Fundus** | APTOS 2019 | **0.922** | **0.858** |
| **OEFI external** (binary DME, OCT) | 1,113 scans | AUC **0.994** · Sens **0.952** · Spec **0.969** |

Checkpoints: `checkpoints/ghota_oct_dme_retfound_sota/`, `checkpoints/ghota_fundus_dr_retfound_sota/`

---

## 1. Unified 3-class DR severity

| Class | Name | Clinical meaning |
|------:|------|------------------|
| **0** | `none` | No DR / no DME |
| **1** | `early_mod` | Mild–moderate (raw grades **1+2+3**) |
| **2** | `advanced` | Severe / advanced (raw grades **4+5**) |

### OCT — DME (MMRDR)

| MMRDR | → GHOTA |
|-------|---------|
| 0 No DME | 0 |
| 1 NCI-DME | 1 |
| 2 CI-DME | 2 |

### Fundus — ICDR-style DR (APTOS)

| ICDR | → GHOTA |
|------|---------|
| 0 | 0 |
| 1, 2, 3 | 1 |
| 4, 5 | 2 |

### Task IDs

| ID | Modality | Role |
|----|----------|------|
| `oct_dme` | OCT B-scan | Trained |
| `fundus_dr` | Colour fundus | Trained (APTOS) |

---

## 2. Why DR-only (not + glaucoma)

- Glaucoma ≠ DR clinically; “advanced” means different things.
- Mixing without separation usually **hurts Acc**.
- GLAUCO covers glaucoma; GHOTA is the clean **DR** track.

---

## 3. Model

```
image ──► backbone (RETFound-OCT or RETFound-CFP)
task_id (oct_dme | fundus_dr) ──► embedding ──► concat ──► 3-class head
```

---

## 4. Data status

| Source | Modality | On disk? | Role |
|--------|----------|----------|------|
| **MMRDR** | OCT | Yes | Primary 3-class |
| **OCTID** | OCT | Yes | Supplement |
| **OEFI** | OCT | Yes | External binary DME |
| **APTOS** | Fundus | Yes | Fundus specialist |
| Paired fundus+OCT | — | **No** | Needed for fused Acc |
| Glaucoma | — | — | **GLAUCO only** |

---

## 5. Fusion

| Technique | Status |
|-----------|--------|
| Dual specialists (task-conditioned) | **Done** |
| OEFI same-eye pairs (409) | **Done** — see `docs/TRAINED_FUSION.md` |
| Late fusion Acc (DR 3-class) | **0.780** avg-eval / **0.802** trained xattn (patient holdout, F1 **0.714**) |
| Infer CLI | `python -m ghota.infer` — route ± fuse |
| OCTA-500 / ROAD | Gated on IEEE login / email (Parth refs) |
| DcardNet data | Private OHSU — cite paper only |

---

## 6. Commands

```powershell
cd C:\Users\HP\OneDrive\Desktop\OCT
.\.venv\Scripts\python.exe scripts/build_ghota_catalogs.py --source all
.\.venv\Scripts\python.exe -m ghota.train --config configs/ghota_oct_dme_retfound_sota.yaml
.\.venv\Scripts\python.exe -m ghota.train --config configs/ghota_fundus_dr_retfound_sota.yaml
.\.venv\Scripts\python.exe scripts/run_ghota_pack.py
```

---

## 7. One-liner

> “GHOTA is our unified **DR** severity model (none / early-moderate / advanced) for OCT and fundus. Glaucoma remains a separate GLAUCO track for accuracy.”
