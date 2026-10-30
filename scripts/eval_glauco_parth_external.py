#!/usr/bin/env python3
"""Run AIROGS RETFound-CFP against hired Parth GLAUCO external fundus sets."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "glauco_retfound_cfp_airogs_light" / "best.pt"
OUT = ROOT / "results" / "glaucoma_parth_external.json"
CAT = ROOT / "data" / "processed" / "glaucoma" / "catalogs"

# Primary hire targets for the supervisor table
PRIMARY = [
    "acrima_external",
    "rim_one_hospital_test",
    "drishti_full",
    "beh_external",
]
# Extra context rows
EXTRA = [
    "rim_one_full",
    "drishti_official_test",
]


def main() -> int:
    if not CKPT.exists():
        print(f"Missing checkpoint: {CKPT}", file=sys.stderr)
        return 1

    # Ensure external catalogs exist
    subprocess.check_call(
        [sys.executable, str(ROOT / "scripts" / "build_parth_external_catalogs.py")],
        cwd=str(ROOT),
    )

    stems = PRIMARY + EXTRA
    csvs = []
    for stem in stems:
        p = CAT / f"{stem}.csv"
        if not p.exists():
            print(f"Missing {p}", file=sys.stderr)
            return 1
        csvs.append(p)

    cmd = [
        sys.executable,
        str(ROOT / "glauco" / "evaluate.py"),
        "--checkpoint",
        str(CKPT),
        "--modality",
        "fundus",
        "--batch-size",
        "16",
        "--image-size",
        "224",
        "--num-workers",
        "2",
        "--out",
        str(OUT),
    ]
    for c in csvs:
        cmd.extend(["--test-csv", str(c)])

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))

    # Compact summary for docs / PPT
    with open(OUT, encoding="utf-8") as f:
        raw = json.load(f)

    summary = {"checkpoint": str(CKPT.relative_to(ROOT)), "hire": {}, "extra": {}}
    for stem in PRIMARY:
        if stem in raw:
            m = raw[stem]
            summary["hire"][stem] = {
                "n": m.get("n") or _n_from_cm(m),
                "auc": m.get("auc"),
                "accuracy": m.get("accuracy"),
                "f1": m.get("f1"),
                "sensitivity": m.get("sensitivity"),
                "specificity": m.get("specificity"),
                "sensitivity_at_95_spec": m.get("sensitivity_at_95_spec"),
            }
    for stem in EXTRA:
        if stem in raw:
            m = raw[stem]
            summary["extra"][stem] = {
                "n": m.get("n") or _n_from_cm(m),
                "auc": m.get("auc"),
                "accuracy": m.get("accuracy"),
                "f1": m.get("f1"),
            }

    summary_path = ROOT / "results" / "glaucoma_parth_external_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")
    return 0


def _n_from_cm(m: dict) -> int | None:
    cm = m.get("confusion_matrix")
    if not cm:
        return None
    return int(sum(sum(r) for r in cm))


if __name__ == "__main__":
    raise SystemExit(main())
