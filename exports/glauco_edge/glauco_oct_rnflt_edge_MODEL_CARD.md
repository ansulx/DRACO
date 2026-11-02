# GLAUCO edge model card — `glauco_efficientnet_harvard_gd_full`

**Product:** glaucoma only (binary screening). Deployment is out of scope; this pack is the peak artifact.

## Architecture & footprint

| Field | Value |
|-------|-------|
| Architecture | `efficientnet_b0` |
| Modality | rnflt |
| Classes | 2 — ['normal', 'glaucoma'] |
| Input | 3×224×224 NCHW |
| Parameters | **4.01 M** (4,010,110) |
| Weights size | **15.458 MB** |
| Pack `.pth` file | 15.586 MB |
| Peak CUDA (1×infer) | 43.13 MB |
| Decision threshold | 0.5 |

## Primary metrics (deploy checklist)

| Metric | Value |
|--------|------:|
| Accuracy | 0.8300 |
| Precision | 0.8864 |
| Recall (=Sensitivity) | 0.7647 |
| Sensitivity | 0.7647 |
| Specificity | 0.8980 |
| F1 | 0.8211 |
| AUC | 0.9130 |
| Sens @ 95% Spec | 0.4314 |

## Splits evaluated

- **harvard_gd_test**: Acc 0.8300 · F1 0.8211 · Sens 0.7647 · Spec 0.8980 · AUC 0.9130
- **train_best_or_val**: Acc 0.8300 · F1 0.8211 · Sens — · Spec — · AUC 0.9130

## Load (deployer)

```python
import torch
pack = torch.load('exports/glauco_edge/glauco_oct_rnflt_edge.pth', map_location='cpu', weights_only=False)
state = pack['state_dict']
classes = pack['class_names']  # ['normal', 'glaucoma']
thr = pack['threshold']
stats = pack['model_stats']
```

Source train checkpoint: `checkpoints/glauco_efficientnet_harvard_gd_full/best.pt`
Exported: 2026-11-02T10:51:12.904389+00:00
