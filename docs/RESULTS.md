# Results

Experiment results for DRACO. Updated after each benchmark run.

---

## Status

| Phase | Status |
|-------|--------|
| Data audit | In progress |
| MMRDR extraction | Pending |
| Baseline training | Not started |
| Foundation model benchmark | Not started |
| External validation (OEFI) | Not started |

---

## Target benchmarks (MMRDR-OCT, DME 3-class)

Published baselines from [MMRDR paper](https://www.nature.com/articles/s41597-026-07005-9) for reference:

| Model | ACC (DME) | F1 (DME) |
|-------|-----------|----------|
| RETFound | 0.897 | 0.759 |
| ResNet-50 | 0.890 | 0.701 |
| ViT | 0.883 | 0.700 |
| KeepFIT | 0.778 | 0.664 |
| FLAIR | 0.728 | 0.559 |

**DRACO goal:** exceed RETFound on MMRDR-OCT test + strong OEFI external performance.

---

## Results table (to be filled)

### Task T1 — MMRDR 3-class DME

| Model | Split | ACC | Macro-F1 | F1 (No) | F1 (NCI) | F1 (CI) | Notes |
|-------|-------|-----|----------|---------|----------|---------|-------|
| — | test | — | — | — | — | — | Pending |

### Task T2 — External validation (OEFI binary DME)

| Model | Train on | Test on | AUC | Acc | Sens@95%Spec | Notes |
|-------|----------|---------|-----|-----|--------------|-------|
| — | MMRDR | OEFI | — | — | — | Pending |

---

## Leaderboard (internal)

```
Rank  Model          MMRDR-F1   OEFI-AUC   Date
----  -----          --------   --------   ----
  -   (no runs yet)
```

---

## How to update this doc

After each experiment:

1. Add row to the relevant table
2. Note config path (e.g. `benchmarks/configs/retfound_mmrdr.yaml`)
3. Link checkpoint if stored locally
4. Brief failure analysis if model underperforms

---

## Known blockers

- [ ] MMRDR not extracted to `data/raw/mmrdr/`
- [ ] Preprocessing pipeline not implemented
- [ ] No GPU training runs yet
