#!/usr/bin/env python3
"""Check whether the A4000 is usable for DRACO training."""

from __future__ import annotations

import shutil
import subprocess
import sys


def run(cmd: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        out = (p.stdout or "") + (p.stderr or "")
        return p.returncode, out.strip()
    except Exception as e:
        return 1, str(e)


def main() -> int:
    print("=== DRACO GPU check ===\n")
    ok = True

    code, out = run(["nvidia-smi"])
    if code != 0 or not out:
        print("FAIL: nvidia-smi not working")
        print("  ", out[:300] if out else "(empty / binary unreadable)")
        print("  Fix: install/update NVIDIA Studio Driver for RTX A4000")
        print("  Download: https://www.nvidia.com/Download/index.aspx")
        print("  Product: RTX A4000 → Studio Driver → Windows 11 64-bit")
        ok = False
    else:
        print("OK: nvidia-smi")
        for line in out.splitlines()[:12]:
            print(" ", line)

    try:
        import torch

        print(f"\ntorch={torch.__version__} cuda_build={torch.version.cuda}")
        if not torch.cuda.is_available():
            print("FAIL: torch.cuda.is_available() == False")
            print("  Common causes:")
            print("  1) Driver too old for this PyTorch CUDA build (need recent Studio/Game Ready)")
            print("  2) nvidia-smi broken / driver install corrupted")
            print("  3) Reboot needed after driver install")
            ok = False
        else:
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"OK: CUDA device 0 = {name} ({mem:.1f} GB)")
            x = torch.randn(1024, 1024, device="cuda")
            y = x @ x
            print(f"OK: matmul test ok, result mean={y.mean().item():.4f}")
    except ImportError:
        print("FAIL: torch not installed — pip install -r requirements.txt")
        ok = False

    print()
    if ok:
        print("GPU ready. Train with:")
        print("  python draco/train.py --config configs/baseline_efficientnet_mmrdr.yaml")
        return 0

    print("GPU NOT ready yet. After fixing the driver:")
    print("  1) Reboot Windows")
    print("  2) python scripts/check_gpu.py")
    print("  3) python draco/train.py --config configs/baseline_efficientnet_mmrdr.yaml")
    return 1


if __name__ == "__main__":
    sys.exit(main())
