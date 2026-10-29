# Trained OEFI paired fusion + infer CLI

Patient-level holdout on OEFI same-eye pairs (train 308 / test 101). Specialists frozen.

## Results (3-class DR)

| Method | Acc | Macro-F1 |
|--------|----:|---------:|
| OCT only | 0.703 | 0.485 |
| Fundus only | 0.653 | 0.372 |
| Avg late fusion (0.5) | 0.733 | 0.527 |
| Learnable late gate | 0.733 | 0.543 |
| **Feature MLP fusion** | **0.762** | **0.635** |
| **Cross-attn fusion** | **0.802** | **0.714** |

Best checkpoint: `checkpoints/ghota_oefi_fusion_xattn/best.pt`  
Produceable default (lighter): `checkpoints/ghota_oefi_fusion_feature/best.pt`

## Infer (production)

```powershell
# route by modality
.\.venv\Scripts\python.exe -m ghota.infer --image <path> --modality oct
.\.venv\Scripts\python.exe -m ghota.infer --image <path> --modality fundus

# paired fuse (real same-eye only)
.\.venv\Scripts\python.exe -m ghota.infer --oct <oct.jpg> --fundus <fundus.jpg> `
  --fusion-ckpt checkpoints/ghota_oefi_fusion_xattn/best.pt --json
```

Train again:

```powershell
.\.venv\Scripts\python.exe -m ghota.train_fusion --config configs/ghota_oefi_fusion_xattn.yaml
```
