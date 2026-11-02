# GLAUCO — peak model pack for edge handoff

**Scope:** glaucoma only (binary: normal vs glaucoma).  
**Our job:** peak Acc + transparent **architecture / params / memory / screening metrics**.  
**Not our job:** on-device deployment (separate team consumes the `.pth`).

## Artifacts

| Pack | Path | Role |
|------|------|------|
| **Peak fundus Acc** | `exports/glauco_edge/glauco_fundus_peak.pth` | RETFound-CFP · AIROGS specialist |
| Model card (JSON) | `exports/glauco_edge/glauco_fundus_peak_model_card.json` | Machine-readable |
| Model card (MD) | `exports/glauco_edge/glauco_fundus_peak_MODEL_CARD.md` | Human readout |
| **Edge-friendly OCT** | `exports/glauco_edge/glauco_oct_rnflt_edge.pth` | EfficientNet-B0 · Harvard-GD RNFLT |

Rebuild:

```powershell
.\.venv\Scripts\python.exe scripts/export_glauco_edge_pack.py `
  --checkpoint checkpoints/glauco_retfound_cfp_airogs_light/best.pt `
  --tag glauco_fundus_peak --out-dir exports/glauco_edge
```

---

## Peak fundus model — footprint

| Field | Value |
|-------|------:|
| Architecture | `retfound_cfp` (ViT-L/16) |
| Classes | 2 — `normal`, `glaucoma` |
| Input | 3×224×224 NCHW |
| Parameters | **303.3 M** |
| Weights | **1157 MB** |
| Pack `.pth` | ~1.1 GB |
| Peak CUDA (1×infer) | **~2334 MB** |
| Threshold | 0.5 |

> Heavy for tiny MCU edges — this is the **accuracy peak**. Deployers may distill/quantize; we ship the teacher Acc + card.

## Peak fundus — primary metrics (AIROGS holdout)

| Metric | Value |
|--------|------:|
| Accuracy | **0.9195** |
| Precision | **0.9195** |
| Recall | **0.9195** |
| Sensitivity | **0.9195** |
| Specificity | **0.9195** |
| F1 | **0.9195** |
| AUC | **0.9687** |
| Sens @ 95% Spec | **0.8286** |

Train-best (same run, val monitor): Acc 0.936 · F1 0.936 · AUC **0.983** · Sens@95spec **0.925**

## External fundus (same peak `.pth`)

| Split | Acc | Sens | Spec | F1 | AUC |
|-------|----:|-----:|-----:|---:|----:|
| Drishti (India) | 0.812 | 0.786 | 0.871 | 0.853 | 0.890 |
| BEH | 0.793 | 0.585 | 0.870 | 0.604 | 0.782 |
| ACRIMA (OD-crop) | 0.452 | 0.045 | 0.974 | 0.085 | 0.578 |
| RIM-ONE hospital | 0.690 | 0.071 | 0.983 | 0.129 | 0.567 |

---

## Edge-friendly OCT companion — footprint

| Field | Value |
|-------|------:|
| Architecture | `efficientnet_b0` |
| Modality | RNFLT heatmap |
| Classes | 2 — `normal`, `glaucoma` |
| Parameters | **4.01 M** |
| Weights | **15.5 MB** |
| Peak CUDA (1×infer) | **~43 MB** |

### Metrics (Harvard-GD test)

| Metric | Value |
|--------|------:|
| Accuracy | 0.830 |
| Precision | 0.886 |
| Recall / Sensitivity | 0.765 |
| Specificity | 0.898 |
| F1 | 0.821 |
| AUC | **0.913** |
| Sens @ 95% Spec | 0.431 |

---

## `.pth` schema (`glauco_edge_v1`)

```text
format_version, task, product, architecture, modality
num_classes, class_names, class_index, input, threshold
model_stats { parameters_*, weights_mb, inference_peak_cuda_mb }
metrics_primary { accuracy, precision, recall, sensitivity, specificity, f1, auc, ... }
metrics_by_split { ... }
config, state_dict
```

Load:

```python
import torch
pack = torch.load("exports/glauco_edge/glauco_fundus_peak.pth", map_location="cpu", weights_only=False)
pack["state_dict"]          # weights
pack["class_names"]         # ['normal', 'glaucoma']
pack["model_stats"]         # params + memory
pack["metrics_primary"]     # screening checklist
```

---

## Metrics checklist we prepare (every pack)

- Accuracy  
- Precision  
- Recall (= sensitivity for positive = glaucoma)  
- Sensitivity  
- Specificity  
- F1  
- AUC (+ AP, Brier when available)  
- Sens @ 85/90/95% Spec + thresholds  
- Confusion matrix  
- Architecture · #params · weights MB · peak infer memory · input shape · #classes  

---

## Note

AIROGS image folder is currently off this machine; holdout numbers for the fundus peak pack are taken from the saved `eval_holdout_test.json` of that run. Externals (Drishti/BEH/ACRIMA/RIM-ONE) were re-evaluated live against the same weights.
