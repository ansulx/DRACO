# GLAUCO edge model card — `glauco_retfound_cfp_airogs_light`

**Product:** glaucoma only (binary screening). Deployment is out of scope; this pack is the peak artifact.

## Architecture & footprint

| Field | Value |
|-------|-------|
| Architecture | `retfound_cfp` |
| Modality | fundus |
| Classes | 2 — ['normal', 'glaucoma'] |
| Input | 3×224×224 NCHW |
| Parameters | **303.304 M** (303,303,682) |
| Weights size | **1157.012 MB** |
| Pack `.pth` file | 1157.13 MB |
| Peak CUDA (1×infer) | 2333.96 MB |
| Decision threshold | 0.5 |

## Primary metrics (deploy checklist)

| Metric | Value |
|--------|------:|
| Accuracy | 0.9195 |
| Precision | 0.9195 |
| Recall (=Sensitivity) | 0.9195 |
| Sensitivity | 0.9195 |
| Specificity | 0.9195 |
| F1 | 0.9195 |
| AUC | 0.9687 |
| Sens @ 95% Spec | 0.8286 |

## Splits evaluated

- **airogs_light_holdout_test**: Acc 0.9195 · F1 0.9195 · Sens 0.9195 · Spec 0.9195 · AUC 0.9687
- **drishti_full**: Acc 0.8119 · F1 0.8527 · Sens 0.7857 · Spec 0.8710 · AUC 0.8899
- **beh_external**: Acc 0.7934 · F1 0.6042 · Sens 0.5848 · Spec 0.8704 · AUC 0.7821
- **acrima_external**: Acc 0.4525 · F1 0.0853 · Sens 0.0455 · Spec 0.9741 · AUC 0.5776
- **rim_one_hospital_test**: Acc 0.6897 · F1 0.1290 · Sens 0.0714 · Spec 0.9831 · AUC 0.5675

## Load (deployer)

```python
import torch
pack = torch.load('exports/glauco_edge/glauco_fundus_peak.pth', map_location='cpu', weights_only=False)
state = pack['state_dict']
classes = pack['class_names']  # ['normal', 'glaucoma']
thr = pack['threshold']
stats = pack['model_stats']
```

Source train checkpoint: `checkpoints/glauco_retfound_cfp_airogs_light/best.pt`
Exported: 2026-11-02T10:50:02.517119+00:00
