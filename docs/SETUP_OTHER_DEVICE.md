# Continue on another device

Git tracks **code, configs, docs, and small result JSONs**.  
It does **not** track raw datasets, catalogs (absolute paths), `.venv`, or model weights.

## 1. Clone

```bash
git clone git@github.com:ansulx/DRACO.git
cd DRACO
git checkout glauco-ar-demo
```

(Use HTTPS if you prefer: `https://github.com/ansulx/DRACO.git`.)

## 2. Python env (Windows example)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\pip install -r requirements.txt huggingface_hub matplotlib
```

Linux/macOS: same packages; use `python3.11 -m venv .venv` and activate accordingly.

## 3. Credentials (local only — never commit)

| Need | Where |
|------|--------|
| Kaggle downloads | `~/.kaggle/kaggle.json` |
| Hugging Face (RETFound) | `HF_TOKEN` env var or `huggingface-cli login` |

## 4. Data + catalogs

Raw data goes under `data/raw/` (gitignored). Rebuild catalogs after download — CSVs use absolute paths.

### GHOTA (DR Acc)

```powershell
# OCT / fundus / OEFI as on the first machine (see docs/SAME_MODALITY_DOWNLOADS.md)
.\.venv\Scripts\python.exe scripts/build_ghota_catalogs.py --source all
.\.venv\Scripts\python.exe scripts/build_oefi_pairs.py
```

### GLAUCO (glaucoma)

```powershell
.\.venv\Scripts\python.exe scripts/download_glaucoma_datasets.py --dataset harvard_gd
# + harvard_gdp / harvard_gf / airogs / Parth fundus sets as needed

.\.venv\Scripts\python.exe scripts/build_glaucoma_catalogs.py --dataset harvard_gd
.\.venv\Scripts\python.exe scripts/build_glaucoma_catalogs.py --dataset acrima
.\.venv\Scripts\python.exe scripts/build_glaucoma_catalogs.py --dataset beh
.\.venv\Scripts\python.exe scripts/build_glaucoma_catalogs.py --dataset rim_one
.\.venv\Scripts\python.exe scripts/build_glaucoma_catalogs.py --dataset biomisa_od
.\.venv\Scripts\python.exe scripts/build_glaucoma_catalogs.py --dataset drishti
```

Parth fundus inventory: [PARTH_GLAUCOMA_FUNDUS.md](PARTH_GLAUCOMA_FUNDUS.md)  
Registry: `data/glaucoma_registry.yaml`

## 5. Weights

```powershell
# RETFound OCT / CFP
.\.venv\Scripts\python.exe scripts/download_retfound_weights.py --modality oct
.\.venv\Scripts\python.exe scripts/download_retfound_weights.py --modality cfp
```

Trained run checkpoints (`checkpoints/**/best.pt`) are **not** in git. Copy them via USB/Drive, or retrain from configs under `configs/`.

## 6. Sanity checks

```powershell
.\.venv\Scripts\python.exe scripts/check_gpu.py
.\.venv\Scripts\python.exe -c "import ghota, glauco, draco; print('ok')"
```

## What lives where

| In GitHub | Local only |
|-----------|------------|
| `draco/`, `ghota/`, `glauco/` | `data/raw/`, `data/processed/` |
| `configs/`, `scripts/`, `docs/` | `checkpoints/*.pt` |
| `data/*_registry.yaml` | `.venv/`, Kaggle/HF secrets |
| Small `results/*.json` | `GLEAM/` dataset folder |
