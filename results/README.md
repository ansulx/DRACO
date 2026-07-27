# Training results (metrics only)

Lightweight copies of training outputs for git tracking.  
**Weights (`.pt`) stay local** in `checkpoints/` (gitignored).

| File | Meaning |
|------|---------|
| `training_index.json` | All runs at a glance |
| `runs/<run_name>/config.yaml` | Exact training config used |
| `runs/<run_name>/best_metrics.json` | Best epoch on MMRDR test |
| `runs/<run_name>/eval_report.json` | MMRDR + OEFI evaluation |

Refresh after new training:

```bash
python scripts/export_training_results.py
```
