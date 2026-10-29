# Same-modality dataset additions (GHOTA)

GHOTA modalities = **colour fundus** + **structural OCT B-scan**.  
**Not** pulled here: OCTA-500 / ROAD (angiography = different modality).

| Dataset | Modality | Status | Role |
|---------|----------|--------|------|
| MMRDR | OCT | On disk | Primary OCT train |
| OCTID | OCT | On disk | OCT supplement |
| OEFI | OCT + fundus (paired) | On disk | External + fusion |
| APTOS | Fundus | On disk | Primary fundus train |
| **IDRiD** | Fundus ICDR | Downloading → `data/raw/idrid/` | Extra fundus Acc / external |
| **Kermany** | OCT | Downloading → `data/raw/kermany/` (~10.8 GB) | Optional OCT scale (DME/CNV/…) |
| OCTA-500 / ROAD | OCTA | Skipped | Different modality |

## Commands

```powershell
.\.venv\Scripts\python.exe scripts/download_ghota_same_modality.py --dataset idrid
.\.venv\Scripts\python.exe scripts/download_ghota_same_modality.py --dataset kermany
.\.venv\Scripts\python.exe scripts/build_ghota_catalogs.py --source all
```

After IDRiD lands, catalogs include `ghota_idrid_*.csv` and `ghota_fundus_all_*.csv` (APTOS+IDRiD).
