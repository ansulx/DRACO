#!/usr/bin/env python3
"""Autonomous GLAUCO training pipeline — runs all jobs on available data.

Trains EfficientNet on Harvard-GDP and combined GD+GDP, cross-evaluates on
held-out sets, updates results index, writes a status log for monitoring.

Usage:
  python scripts/run_glaucoma_autopilot.py
  python scripts/run_glaucoma_autopilot.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

STATUS_PATH = ROOT / "results" / "autopilot_status.json"
LOG_DIR = ROOT / "results" / "autopilot_logs"
INDEX_PATH = ROOT / "results" / "glaucoma_training_index.json"

TRAIN_CONFIGS = [
    {
        "id": "gdp_full",
        "config": "configs/glaucoma_efficientnet_harvard_gdp_full.yaml",
        "run_name": "glauco_efficientnet_harvard_gdp_full",
        "dataset": "harvard_gdp",
        "train_n": 800,
    },
    {
        "id": "combined_full",
        "config": "configs/glaucoma_efficientnet_harvard_combined_full.yaml",
        "run_name": "glauco_efficientnet_harvard_combined_full",
        "dataset": "harvard_combined",
        "train_n": 1200,
    },
]

CROSS_EVALS = [
    {
        "checkpoint_run": "glauco_efficientnet_harvard_gdp_full",
        "test_csv": "data/processed/glaucoma/catalogs/harvard_gd_test.csv",
        "label": "gdp_model_on_gd_test",
    },
    {
        "checkpoint_run": "glauco_efficientnet_harvard_combined_full",
        "test_csv": "data/processed/glaucoma/catalogs/harvard_gdp_test.csv",
        "label": "combined_model_on_gdp_test",
    },
    {
        "checkpoint_run": "glauco_efficientnet_harvard_combined_full",
        "test_csv": "data/processed/glaucoma/catalogs/harvard_gd_test.csv",
        "label": "combined_model_on_gd_test",
    },
]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_status(state: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = _now()
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def run_cmd(cmd: list[str], log_path: Path, dry_run: bool = False) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = " ".join(cmd)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n[{_now()}] {line}\n")
    if dry_run:
        print(f"[dry-run] {line}")
        return 0
    with open(log_path, "a", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return proc.returncode


def merge_index(new_entries: list[dict]) -> None:
  entries = []
  if INDEX_PATH.exists():
      with open(INDEX_PATH, encoding="utf-8") as f:
          entries = json.load(f)
  by_name = {e["run_name"]: e for e in entries}
  for e in new_entries:
      by_name[e["run_name"]] = e
  with open(INDEX_PATH, "w", encoding="utf-8") as f:
      json.dump(list(by_name.values()), f, indent=2)


def load_metrics(run_name: str) -> dict | None:
    p = ROOT / "checkpoints" / run_name / "best_metrics.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="GLAUCO autopilot")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_log = LOG_DIR / f"autopilot_{run_id}.log"

    state: dict = {
        "run_id": run_id,
        "started_at": _now(),
        "phase": "starting",
        "jobs": [],
        "cross_evals": [],
        "errors": [],
    }
    write_status(state)

    try:
        # Phase 1: catalogs
        state["phase"] = "build_catalogs"
        write_status(state)
        for ds in ("harvard_gd", "harvard_gdp", "harvard_combined"):
            rc = run_cmd(
                [str(PYTHON), "scripts/build_glaucoma_catalogs.py", "--dataset", ds],
                master_log,
                args.dry_run,
            )
            if rc != 0:
                raise RuntimeError(f"catalog build failed: {ds}")

        # Phase 2: training
        for job in TRAIN_CONFIGS:
            state["phase"] = f"training_{job['id']}"
            write_status(state)
            job_state = {"id": job["id"], "status": "running", "started_at": _now()}
            state["jobs"].append(job_state)
            write_status(state)

            t0 = time.time()
            rc = run_cmd(
                [str(PYTHON), "glauco/train.py", "--config", job["config"]],
                master_log,
                args.dry_run,
            )
            elapsed = round(time.time() - t0, 1)
            metrics = load_metrics(job["run_name"]) if not args.dry_run else None

            job_state.update(
                {
                    "status": "ok" if rc == 0 else "failed",
                    "elapsed_s": elapsed,
                    "metrics": metrics,
                    "finished_at": _now(),
                }
            )
            write_status(state)

            if rc != 0:
                state["errors"].append(f"train failed: {job['id']}")
                write_status(state)
                continue

            if metrics:
                merge_index(
                    [
                        {
                            "run_name": job["run_name"],
                            "track": "oct_rnflt",
                            "model": "efficientnet_b0",
                            "dataset": job["dataset"],
                            "best_epoch": metrics.get("epoch"),
                            "metrics": {
                                k: metrics[k]
                                for k in (
                                    "accuracy",
                                    "macro_f1",
                                    "auc",
                                    "sensitivity_at_95_spec",
                                )
                                if k in metrics
                            },
                            "autopilot_run": run_id,
                        }
                    ]
                )

        # Phase 3: cross-dataset evaluation
        state["phase"] = "cross_eval"
        write_status(state)
        for ev in CROSS_EVALS:
            ckpt = ROOT / "checkpoints" / ev["checkpoint_run"] / "best.pt"
            if not ckpt.exists() and not args.dry_run:
                state["cross_evals"].append(
                    {"label": ev["label"], "status": "skipped", "reason": "no checkpoint"}
                )
                write_status(state)
                continue

            out = ROOT / "checkpoints" / ev["checkpoint_run"] / f"eval_{ev['label']}.json"
            rc = run_cmd(
                [
                    str(PYTHON),
                    "glauco/evaluate.py",
                    "--checkpoint",
                    str(ckpt),
                    "--test-csv",
                    ev["test_csv"],
                    "--modality",
                    "rnflt",
                    "--out",
                    str(out),
                ],
                master_log,
                args.dry_run,
            )
            ev_result = {"label": ev["label"], "status": "ok" if rc == 0 else "failed"}
            if out.exists():
                with open(out, encoding="utf-8") as f:
                    ev_result["report"] = json.load(f)
            state["cross_evals"].append(ev_result)
            write_status(state)

        # Phase 4: optional RETFound-CFP if HF token present
        import os

        if os.environ.get("HF_TOKEN") and not args.dry_run:
            state["phase"] = "download_retfound_cfp"
            write_status(state)
            cfp = ROOT / "checkpoints" / "weights" / "RETFound_cfp.pth"
            if not cfp.exists():
                run_cmd(
                    [
                        str(PYTHON),
                        "scripts/download_retfound_weights.py",
                        "--modality",
                        "cfp",
                    ],
                    master_log,
                )

        state["phase"] = "complete"
        state["finished_at"] = _now()
        write_status(state)
        print(f"Autopilot complete. Status: {STATUS_PATH}")
        print(f"Log: {master_log}")
        return 0

    except Exception as e:
        state["phase"] = "error"
        state["errors"].append(str(e))
        state["traceback"] = traceback.format_exc()
        state["finished_at"] = _now()
        write_status(state)
        print(f"Autopilot error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
