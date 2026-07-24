# Make the A4000 available for DRACO training

PyTorch is already installed (`torch 2.6.0+cu124`). Training fails because the **NVIDIA driver is broken/outdated**, not because the GPU is missing.

## Diagnosis (this machine)

| Check | Result |
|-------|--------|
| GPU in Device Manager | NVIDIA RTX A4000 — OK |
| Driver version | `27.21.14.6296` (~**462.96**, from ~2021) |
| `nvidia-smi.exe` | **corrupted / unreadable** |
| `torch.cuda.is_available()` | **False** |

PyTorch cu124 needs a **recent** driver (roughly 550+). Your driver is far too old and `nvidia-smi` itself is damaged.

## Fix (you must run the installer — it needs admin + reboot)

### Option A — installer already downloading

File (when download finishes):

`C:\Users\HP\Downloads\NVIDIA_Studio_610.62.exe`

1. Right-click → **Run as administrator**
2. Express / custom install (clean install if offered)
3. **Reboot**
4. Verify:

```powershell
nvidia-smi
cd "C:\Users\HP\OneDrive\Desktop\OCT"
python scripts\check_gpu.py
```

### Option B — download from NVIDIA

1. Open: https://www.nvidia.com/Download/index.aspx  
2. Product type: **NVIDIA RTX / Quadro** → **RTX A4000**  
3. OS: **Windows 11 64-bit**  
4. Download type: **Studio Driver** (preferred) or Game Ready  
5. Install as admin → reboot

### Then train on full MMRDR

```powershell
cd "C:\Users\HP\OneDrive\Desktop\OCT"
python scripts\check_gpu.py
python draco\train.py --config configs\baseline_efficientnet_mmrdr.yaml
python draco\evaluate.py --checkpoint checkpoints\efficientnet_b0_mmrdr\best.pt
```

## What I cannot do from here

- Driver install needs **admin UAC** and a **reboot** — Cursor cannot finish that alone.
- After reboot, CUDA will light up and full A4000 training works with the configs already in the repo.

## RETFound weights (separate)

Also gated on Hugging Face. After GPU works:

1. Accept access: https://huggingface.co/YukunZhou/RETFound_mae_natureOCT  
2. `huggingface-cli login`  
3. Download OCT `.pth` → set `model.checkpoint` in `configs/retfound_mmrdr.yaml`
