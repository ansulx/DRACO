#!/usr/bin/env python3
"""Long unattended GLAUCO improvement pack (A4000).

Trains extra ablations + stronger tri-modal variants, evaluates, refreshes
figures/docs. Safe to leave running for hours. Skips runs that already have
best.pt unless --force.

Usage:
  .\\.venv\\Scripts\\python.exe -u scripts/run_glaucoma_long_pack.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not PY.exists():
    PY = Path(sys.executable)

CAT = ROOT / "data" / "processed" / "glaucoma" / "catalogs"
LOG_DIR = ROOT / "results" / "autopilot_logs"
FIG = ROOT / "results" / "glaucoma_figures"
DOC = ROOT / "docs" / "GLAUCOMA_RESULTS_2026.md"
PACK = ROOT / "results" / "glaucoma_long_pack.json"

# Ordered longest / highest-value first-ish; script is sequential either way.
CONFIGS = [
    "configs/glaucoma_gleam_vf_eff.yaml",
    "configs/glaucoma_gleam_fundus_eff.yaml",
    "configs/glaucoma_gleam_oct_eff.yaml",
    "configs/glaucoma_fusion_gleam_trimodal_binary.yaml",
    "configs/glaucoma_fusion_gleam_trimodal_xattn_v2.yaml",
]

HAMM = {"accuracy": 0.8108, "macro_f1": 0.7590}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str]) -> int:
    log("CMD " + " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


def ensure_catalogs() -> None:
    if not (CAT / "gleam_paired_train.csv").exists():
        run([str(PY), "scripts/build_glaucoma_catalogs.py", "--dataset", "gleam"])
    for split in ("train", "val", "test"):
        src = CAT / f"gleam_paired_{split}.csv"
        if not src.exists():
            continue
        df = pd.read_csv(src)
        for col, name in (
            ("oct_path", "gleam_oct"),
            ("fundus_path", "gleam_fundus"),
            ("vf_path", "gleam_vf"),
        ):
            out = CAT / f"{name}_{split}.csv"
            part = df[["label", "label_name", "split", "dataset"]].copy()
            part["path"] = df[col]
            part["modality"] = name.replace("gleam_", "")
            part.to_csv(out, index=False)
    log("Catalogs ready")


def train_one(cfg_rel: str, force: bool) -> str | None:
    cfg_path = ROOT / cfg_rel
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    name = cfg["run_name"]
    ckpt = ROOT / "checkpoints" / name / "best.pt"
    if ckpt.exists() and not force:
        log(f"SKIP exists: {name}")
        return name
    rc = run([str(PY), "-u", "-m", "glauco.train", "--config", str(cfg_path)])
    if rc != 0:
        log(f"FAIL train {name} rc={rc}")
        return name if ckpt.exists() else None
    return name


def eval_one(name: str) -> dict | None:
    ckpt = ROOT / "checkpoints" / name / "best.pt"
    if not ckpt.exists():
        return None
    # pick matching test csv
    cfg_yaml = ROOT / "checkpoints" / name / "config.yaml"
    test_csv = CAT / "gleam_paired_test.csv"
    if cfg_yaml.exists():
        with open(cfg_yaml, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        train_csv = str((cfg.get("data") or {}).get("train_csv", ""))
        if "gleam_oct" in train_csv:
            test_csv = CAT / "gleam_oct_test.csv"
        elif "gleam_fundus" in train_csv:
            test_csv = CAT / "gleam_fundus_test.csv"
        elif "gleam_vf" in train_csv:
            test_csv = CAT / "gleam_vf_test.csv"
    out = ROOT / "checkpoints" / name / "eval_official_test.json"
    rc = run(
        [
            str(PY),
            "-u",
            "-m",
            "glauco.evaluate",
            "--checkpoint",
            str(ckpt),
            "--test-csv",
            str(test_csv),
            "--batch-size",
            "8",
            "--out",
            str(out),
        ]
    )
    if rc != 0 or not out.exists():
        log(f"FAIL eval {name}")
        return None
    report = json.loads(out.read_text(encoding="utf-8"))
    for v in report.values():
        if isinstance(v, dict) and ("accuracy" in v or "auc" in v):
            return v
    return None


def refresh_outputs(results: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, t in results.items():
        if not t:
            continue
        rows.append(
            {
                "run": name.replace("glauco_", "").replace("fusion_gleam_", ""),
                "accuracy": t.get("accuracy"),
                "macro_f1": t.get("macro_f1"),
                "auc": t.get("auc") or (t.get("binary_remapped") or {}).get("auc"),
            }
        )
    if rows:
        df = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_df = df.dropna(subset=["macro_f1"]).sort_values("macro_f1")
        if len(plot_df):
            x = range(len(plot_df))
            ax.bar([i - 0.2 for i in x], plot_df["accuracy"], 0.4, label="Acc")
            ax.bar([i + 0.2 for i in x], plot_df["macro_f1"], 0.4, label="Macro-F1")
            ax.axhline(HAMM["accuracy"], color="#c44", ls="--", label="HAMM Acc")
            ax.axhline(HAMM["macro_f1"], color="#e80", ls=":", label="HAMM F1")
            ax.set_xticks(list(x))
            ax.set_xticklabels(plot_df["run"], rotation=30, ha="right", fontsize=8)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8)
            ax.set_title("Long-pack GLEAM results vs HAMM 2026")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(FIG / "long_pack_gleam_vs_hamm.png", dpi=160)
            plt.close(fig)

    PACK.write_text(json.dumps({"finished": datetime.now().isoformat(), "results": results}, indent=2), encoding="utf-8")

    fixed = [
        "",
        "## 7. Long improvement pack (auto)",
        "",
        f"Finished: {datetime.now().isoformat(timespec='minutes')}",
        "",
        "| Run | Acc | Macro-F1 | AUC / binary AUC |",
        "|-----|-----|----------|------------------|",
    ]
    for name, t in results.items():
        if not t:
            fixed.append(f"| `{name}` | — | — | failed/missing |")
            continue
        acc = t.get("accuracy")
        f1 = t.get("macro_f1")
        auc = t.get("auc") or (t.get("binary_remapped") or {}).get("auc")
        acc_s = f"{acc:.3f}" if acc is not None else "—"
        f1_s = f"{f1:.3f}" if f1 is not None else "—"
        auc_s = f"{auc:.3f}" if auc is not None else "—"
        fixed.append(f"| `{name}` | {acc_s} | {f1_s} | {auc_s} |")
    fixed += ["", "Figure: `results/glaucoma_figures/long_pack_gleam_vs_hamm.png`", ""]

    if DOC.exists():
        text = DOC.read_text(encoding="utf-8")
        if "## 7. Long improvement pack" in text:
            text = text.split("## 7. Long improvement pack")[0].rstrip() + "\n"
        DOC.write_text(text + "\n".join(fixed), encoding="utf-8")
    else:
        DOC.write_text("\n".join(fixed), encoding="utf-8")
    log(f"Updated {DOC}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Long pack start | force={args.force} | python={PY}")
    ensure_catalogs()

    names: list[str] = []
    for cfg in CONFIGS:
        try:
            n = train_one(cfg, force=args.force)
            if n:
                names.append(n)
        except Exception:
            log(traceback.format_exc())

    # Also re-eval prior best fusion checkpoints for the table
    for extra in (
        "glauco_fusion_gleam_trimodal_xattn",
        "glauco_fusion_gleam_trimodal_late",
        "glauco_fusion_gleam_dual_xattn_eff",
    ):
        if (ROOT / "checkpoints" / extra / "best.pt").exists() and extra not in names:
            names.append(extra)

    results: dict = {}
    for name in names:
        try:
            results[name] = eval_one(name)
        except Exception:
            log(traceback.format_exc())
            results[name] = None

    refresh_outputs(results)
    log("LONG PACK COMPLETE")
    log(f"JSON: {PACK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
