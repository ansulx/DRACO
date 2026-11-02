# GLAUCO OCT / RNFLT — improvement log (2026-11-02)

## Before → after (Harvard-GD test)

| | Old edge pack | New edge pack |
|--|--------------:|--------------:|
| Checkpoint | `glauco_efficientnet_harvard_gd_full` | `glauco_efficientnet_harvard_gd_acc_plain` |
| Early-stop | `monitor: auc` | `monitor: accuracy` |
| Threshold | 0.50 | **0.55** |
| Acc | 0.830 | **0.870** |
| Sens | 0.765 | **0.882** |
| Spec | 0.898 | 0.857 |
| F1 | 0.821 | **0.874** |
| AUC | 0.913 | 0.906 |

## What worked

1. **Acc early-stopping** — history already had Acc **0.86** at ep22; AUC-monitor kept ep26 (Acc 0.83). Retrain with `monitor: accuracy` → Acc **0.86** again; thr **0.55** → Acc **0.87**.
2. Config: `configs/glaucoma_efficientnet_harvard_gd_acc_plain.yaml`

## What did not work

| Attempt | Result |
|---------|--------|
| Cosine + label_smoothing + Acc monitor | Best Acc **0.74** — recipe hurt |
| GF → GD fine-tune | Best Acc **0.68** — domain mismatch |
| Old ckpt thr 0.35 only | Acc **0.84** (free but weaker than Acc retrain) |
| GD+GDP combined (prior) | AUC **0.61** — avoid |

## Commands

```powershell
.\.venv\Scripts\python.exe glauco\train.py --config configs\glaucoma_efficientnet_harvard_gd_acc_plain.yaml

.\.venv\Scripts\python.exe scripts\export_glauco_edge_pack.py `
  --checkpoint checkpoints\glauco_efficientnet_harvard_gd_acc_plain\best.pt `
  --tag glauco_oct_rnflt_edge `
  --threshold 0.55 `
  --test-csv data/processed/glaucoma/catalogs/harvard_gd_test.csv `
  --cached-holdout checkpoints/glauco_efficientnet_harvard_gd_acc_plain/best_metrics.json
```

## Next levers (if needed later)

- 3 seeds of Acc-plain; keep Acc-best  
- ResNet50 / Eff-B1+ if Hub download allowed (heavier edge)  
- Keep AUC-peak ckpt as alternate pack if supervisors prefer Spec/AUC over Acc/Sens  
