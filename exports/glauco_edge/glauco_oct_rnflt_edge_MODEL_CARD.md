# GLAUCO edge model card — `glauco_efficientnet_harvard_gd_acc_plain`

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
| Pack `.pth` file | 15.587 MB |
| Peak CUDA (1×infer) | 43.13 MB |
| Decision threshold | 0.55 |

## Primary metrics (deploy checklist)

| Metric | Value |
|--------|------:|
| Accuracy | 0.8700 |
| Precision | 0.8654 |
| Recall (=Sensitivity) | 0.8824 |
| Sensitivity | 0.8824 |
| Specificity | 0.8571 |
| F1 | 0.8738 |
| AUC | 0.9064 |
| Sens @ 95% Spec | 0.3725 |

## Splits evaluated

- **harvard_gd_test**: Acc 0.8700 · F1 0.8738 · Sens 0.8824 · Spec 0.8571 · AUC 0.9064
- **train_best_or_val**: Acc 0.8600 · F1 0.8654 · Sens 0.8824 · Spec 0.8367 · AUC 0.9064

## Load (deployer)

```python
import torch
pack = torch.load('exports/glauco_edge/glauco_oct_rnflt_edge.pth', map_location='cpu', weights_only=False)
state = pack['state_dict']
classes = pack['class_names']  # ['normal', 'glaucoma']
thr = pack['threshold']
stats = pack['model_stats']
```

Source train checkpoint: `checkpoints/glauco_efficientnet_harvard_gd_acc_plain/best.pt`
Exported: 2026-11-02T11:39:40.155756+00:00
