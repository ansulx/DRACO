#!/usr/bin/env python3
"""Build GLAUCO *external* catalogs for hired Parth fundus sets.

External = never used for AIROGS training. Prefer official OOD splits where they
exist (RIM-ONE by-hospital test, Drishti Test); otherwise use the full set.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CAT = ROOT / "data" / "processed" / "glaucoma" / "catalogs"
OUT = CAT
STATS = ROOT / "data" / "processed" / "glaucoma" / "stats"


def _load(name: str) -> pd.DataFrame:
    return pd.read_csv(CAT / name)


def _write(df: pd.DataFrame, stem: str) -> dict:
    df = df.copy()
    df["split"] = "external"
    path = OUT / f"{stem}.csv"
    df.to_csv(path, index=False)
    n = len(df)
    g = int(df["label"].sum())
    info = {"csv": str(path.relative_to(ROOT)), "n": n, "glaucoma": g, "normal": n - g}
    print(f"{stem}: n={n} G={g} N={n - g} -> {path.name}")
    return info


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}

    # ACRIMA / BEH: full set (random train/test was only for if we ever fine-tune)
    report["acrima_external"] = _write(
        pd.concat([_load("acrima_train.csv"), _load("acrima_test.csv")], ignore_index=True),
        "acrima_external",
    )
    report["beh_external"] = _write(
        pd.concat([_load("beh_train.csv"), _load("beh_test.csv")], ignore_index=True),
        "beh_external",
    )

    # RIM-ONE: official by-hospital TEST is the real OOD hire target
    report["rim_one_hospital_test"] = _write(_load("rim_one_test.csv"), "rim_one_hospital_test")
    report["rim_one_full"] = _write(
        pd.concat([_load("rim_one_train.csv"), _load("rim_one_test.csv")], ignore_index=True),
        "rim_one_full",
    )

    # Drishti: official Test + full (Indian story)
    report["drishti_official_test"] = _write(_load("drishti_test.csv"), "drishti_official_test")
    report["drishti_full"] = _write(
        pd.concat([_load("drishti_train.csv"), _load("drishti_test.csv")], ignore_index=True),
        "drishti_full",
    )

    STATS.mkdir(parents=True, exist_ok=True)
    meta = STATS / "parth_external_catalogs.json"
    with open(meta, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {meta}")


if __name__ == "__main__":
    main()
