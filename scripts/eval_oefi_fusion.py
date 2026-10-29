"""Evaluate GHOTA OCT/fundus specialists + late fusion on OEFI paired eyes.

Targets:
  - binary DME (label_dme) — primary OEFI external task
  - optional 3-class DR when label_dr_ghota is present
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from draco.metrics import binary_metrics, multiclass_metrics
from ghota.models.factory import build_ghota_from_config
from ghota.models.fusion import LateFusion
from glauco.data.datasets import load_image
from glauco.data.transforms import build_transforms

OUT = ROOT / "docs" / "ghota"


class PairedOefiDataset(Dataset):
    def __init__(self, csv: Path, image_size: int = 224):
        self.df = pd.read_csv(csv)
        self.tf_o = build_transforms("eval", image_size, "oct")
        self.tf_f = build_transforms("eval", image_size, "fundus")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        xo = self.tf_o(load_image(Path(row["path_oct"]), "oct"))
        xf = self.tf_f(load_image(Path(row["path_fundus"]), "fundus"))
        y_dme = torch.tensor(int(row["label_dme"]), dtype=torch.long)
        dr = row.get("label_dr_ghota")
        y_dr = torch.tensor(-1 if pd.isna(dr) else int(dr), dtype=torch.long)
        return xo, xf, y_dme, y_dr


def _load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config") or yaml.safe_load((ckpt_path.parent / "config.yaml").read_text(encoding="utf-8"))
    model = build_ghota_from_config(cfg, load_weights=False).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    disease = list(cfg.get("diseases") or ["oct_dme"])[0]
    # disease id is always 0 when single-disease config
    return model, 0, disease


@torch.no_grad()
def run(args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = PairedOefiDataset(args.pairs, image_size=args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model_o, did_o, _ = _load_model(args.oct_ckpt, device)
    model_f, did_f, _ = _load_model(args.fundus_ckpt, device)
    fusion = LateFusion(num_classes=3, learnable=False).to(device)

    ys_dme, p_o, p_f, p_fuse = [], [], [], []
    ys_dr, pred_o3, pred_f3, pred_fuse3 = [], [], [], []

    for xo, xf, y_dme, y_dr in loader:
        xo = xo.to(device, non_blocking=True)
        xf = xf.to(device, non_blocking=True)
        d_o = torch.full((xo.size(0),), did_o, device=device, dtype=torch.long)
        d_f = torch.full((xf.size(0),), did_f, device=device, dtype=torch.long)
        lo = model_o(xo, d_o)
        lf = model_f(xf, d_f)
        # DME binary score = P(class>=1) = 1 - P(0)
        po = F.softmax(lo, dim=1)
        pf = F.softmax(lf, dim=1)
        fuse = fusion(lo, lf)  # probs (default)
        p_o.append((1.0 - po[:, 0]).cpu().numpy())
        p_f.append((1.0 - pf[:, 0]).cpu().numpy())
        p_fuse.append((1.0 - fuse[:, 0]).cpu().numpy())
        ys_dme.append(y_dme.numpy())

        pred_o3.append(po.argmax(1).cpu().numpy())
        pred_f3.append(pf.argmax(1).cpu().numpy())
        pred_fuse3.append(fuse.argmax(1).cpu().numpy())
        ys_dr.append(y_dr.numpy())

    y_dme = np.concatenate(ys_dme)
    report = {
        "n_pairs": int(len(y_dme)),
        "dme_prevalence": float(y_dme.mean()),
        "oct_only_dme": binary_metrics(y_dme, np.concatenate(p_o)),
        "fundus_only_dme": binary_metrics(y_dme, np.concatenate(p_f)),
        "late_fusion_dme": binary_metrics(y_dme, np.concatenate(p_fuse)),
    }

    y_dr = np.concatenate(ys_dr)
    mask = y_dr >= 0
    if mask.sum() >= 10:
        report["dr_3class_n"] = int(mask.sum())
        report["oct_only_dr"] = multiclass_metrics(y_dr[mask], np.concatenate(pred_o3)[mask])
        report["fundus_only_dr"] = multiclass_metrics(y_dr[mask], np.concatenate(pred_f3)[mask])
        report["late_fusion_dr"] = multiclass_metrics(y_dr[mask], np.concatenate(pred_fuse3)[mask])

    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs",
        type=Path,
        default=ROOT / "data/processed/ghota/catalogs/ghota_oefi_paired.csv",
    )
    parser.add_argument(
        "--oct-ckpt",
        type=Path,
        default=ROOT / "checkpoints/ghota_oct_dme_retfound_sota/best.pt",
    )
    parser.add_argument(
        "--fundus-ckpt",
        type=Path,
        default=ROOT / "checkpoints/ghota_fundus_dr_retfound_sota/best.pt",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    if not args.pairs.exists():
        print(f"Missing pairs CSV: {args.pairs}. Run scripts/build_oefi_pairs.py first.")
        return 1

    report = run(args)
    OUT.mkdir(parents=True, exist_ok=True)
    out_json = OUT / "oefi_paired_fusion_eval.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out_json}")

    # short markdown append-friendly summary
    md = OUT.parent / "OEFI_PAIRED_FUSION.md"
    lines = [
        "# OEFI paired late fusion (GHOTA specialists)",
        "",
        f"Pairs: **{report['n_pairs']}** same-eye fundus+OCT from "
        "[OEFI](https://github.com/Traslational-Visual-Health-Laboratory/OCT-AND-EYE-FUNDUS-DATASET).",
        "",
        "## Binary DME",
        "",
        "| Branch | AUC | Sens | Spec | Acc | F1 |",
        "|--------|----:|-----:|-----:|----:|---:|",
    ]
    for key, name in [
        ("oct_only_dme", "OCT specialist"),
        ("fundus_only_dme", "Fundus specialist"),
        ("late_fusion_dme", "**Late fusion**"),
    ]:
        m = report[key]
        lines.append(
            f"| {name} | {m.get('auc', 0):.3f} | {m.get('sensitivity', 0):.3f} | "
            f"{m.get('specificity', 0):.3f} | {m.get('accuracy', 0):.3f} | {m.get('f1', 0):.3f} |"
        )
    if "late_fusion_dr" in report:
        lines += [
            "",
            f"## 3-class DR (n={report['dr_3class_n']} labeled)",
            "",
            "| Branch | Acc | Macro-F1 |",
            "|--------|----:|---------:|",
            f"| OCT | {report['oct_only_dr']['accuracy']:.3f} | {report['oct_only_dr']['macro_f1']:.3f} |",
            f"| Fundus | {report['fundus_only_dr']['accuracy']:.3f} | {report['fundus_only_dr']['macro_f1']:.3f} |",
            f"| Late fusion | {report['late_fusion_dr']['accuracy']:.3f} | {report['late_fusion_dr']['macro_f1']:.3f} |",
        ]
    lines += ["", f"JSON: `{out_json.as_posix()}`", ""]
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
