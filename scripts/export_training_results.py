#!/usr/bin/env python3
"""Copy training metrics from checkpoints/ to results/runs/ for git tracking."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints"
OUT = ROOT / "results" / "runs"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    index: list[dict] = []

    if not CKPT.exists():
        print("No checkpoints/ directory")
        return

    for run_dir in sorted(p for p in CKPT.iterdir() if p.is_dir() and p.name != "weights"):
        name = run_dir.name
        dest = OUT / name
        dest.mkdir(parents=True, exist_ok=True)
        entry: dict = {"run_name": name, "files": []}

        for fname in ("config.yaml", "best_metrics.json", "eval_report.json"):
            src = run_dir / fname
            if src.exists():
                shutil.copy2(src, dest / fname)
                entry["files"].append(fname)

        bm = run_dir / "best_metrics.json"
        if bm.exists():
            m = json.loads(bm.read_text(encoding="utf-8"))
            entry["best_epoch"] = m.get("epoch")
            entry["mmrdr"] = {
                "accuracy": m.get("accuracy"),
                "macro_f1": m.get("macro_f1"),
                "per_class_f1": m.get("per_class_f1"),
            }

        ev = run_dir / "eval_report.json"
        if ev.exists():
            e = json.loads(ev.read_text(encoding="utf-8"))
            entry["oefi"] = e.get("oefi")

        cfg = run_dir / "config.yaml"
        if cfg.exists():
            try:
                import yaml

                c = yaml.safe_load(cfg.read_text(encoding="utf-8"))
                entry["config_summary"] = {
                    "model": c.get("model", {}).get("name"),
                    "batch_size": c.get("batch_size"),
                    "epochs": c.get("epochs"),
                    "loss": c.get("loss", "ce"),
                    "blr": c.get("blr"),
                    "lr": c.get("lr"),
                }
            except ImportError:
                entry["config_summary"] = {"config_yaml": str(cfg.relative_to(ROOT))}

        index.append(entry)

    idx_path = ROOT / "results" / "training_index.json"
    idx_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Exported {len(index)} runs -> results/runs/")
    for e in index:
        f1 = (e.get("mmrdr") or {}).get("macro_f1")
        print(f"  {e['run_name']}: macro_f1={f1}")


if __name__ == "__main__":
    main()
