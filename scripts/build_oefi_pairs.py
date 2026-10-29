"""Build OEFI patient/eye-paired fundus+OCT catalogs for GHOTA late fusion.

Pairing key from OEFI nomenclature: ``{patient}_{OD|OI}``
(filenames: ``1221_OD_f_3.jpg`` fundus, ``1221_OD_o_1.jpg`` OCT).

Labels: prefer OCT DME (binary) as fusion target when both exist; also keep DR strings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OEFI = ROOT / "data" / "raw" / "oefi"
OUT = ROOT / "data" / "processed" / "ghota" / "catalogs"


def _stem_key(name: str) -> tuple[str, str, str, str] | None:
    # 1221_OD_f_3.jpg -> patient, eye, modality, sample
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    return parts[0], parts[1], parts[2], parts[3]


def _index_images(folder: Path, modality_code: str) -> dict[tuple[str, str], list[Path]]:
    """Map (patient, eye) -> list of image paths for modality o|f."""
    out: dict[tuple[str, str], list[Path]] = {}
    if not folder.exists():
        return out
    for p in folder.rglob("*.jpg"):
        parsed = _stem_key(p.name)
        if parsed is None:
            continue
        patient, eye, mod, _sample = parsed
        if mod != modality_code:
            continue
        out.setdefault((patient, eye), []).append(p)
    return out


def _load_meta(csv_path: Path, path_col_candidates: tuple[str, ...] = ("image", "Image", "filename", "file")) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # normalize columns
    cols = {c.lower().strip(): c for c in df.columns}
    dme_col = cols.get("dme")
    dr_col = cols.get("dr")
    # find a name column
    name_col = None
    for cand in path_col_candidates + tuple(df.columns):
        key = cand.lower() if isinstance(cand, str) else ""
        if key in cols and any(x in key for x in ("name", "file", "image", "path", "id")):
            name_col = cols[key]
            break
    if name_col is None:
        # first column often is the filename stem
        name_col = df.columns[0]
    out = pd.DataFrame(
        {
            "raw_name": df[name_col].astype(str),
            "dme": df[dme_col].astype(int) if dme_col else None,
            "dr": df[dr_col].astype(str) if dr_col else None,
        }
    )
    keys = []
    for n in out["raw_name"]:
        # may be path or stem
        base = Path(n).name
        parsed = _stem_key(base if "_" in base else f"{base}.jpg")
        if parsed is None:
            # try attaching dummy
            parsed = _stem_key(base + ".jpg") if not base.endswith(".jpg") else None
        if parsed is None:
            keys.append((None, None))
        else:
            keys.append((parsed[0], parsed[1]))
    out["patient"] = [k[0] for k in keys]
    out["eye"] = [k[1] for k in keys]
    return out.dropna(subset=["patient", "eye"])


def build_pairs() -> pd.DataFrame:
    fundus_dir = OEFI / "EYE FUNDUS"
    oct_dir = OEFI / "OCT"
    if not fundus_dir.exists():
        raise FileNotFoundError(
            f"Missing {fundus_dir}. Run: python scripts/download_datasets.py --dataset oefi --full"
        )
    f_idx = _index_images(fundus_dir, "f")
    o_idx = _index_images(oct_dir, "o")
    print(f"Fundus eyes={len(f_idx)} OCT eyes={len(o_idx)}")

    oct_meta = _load_meta(OEFI / "OCT.csv") if (OEFI / "OCT.csv").exists() else pd.DataFrame()
    fun_meta = _load_meta(OEFI / "EYE FUNDUS.csv") if (OEFI / "EYE FUNDUS.csv").exists() else pd.DataFrame()

    meta_o = {}
    if len(oct_meta):
        for _, r in oct_meta.iterrows():
            meta_o[(r["patient"], r["eye"])] = {"dme": r["dme"], "dr": r["dr"]}
    meta_f = {}
    if len(fun_meta):
        for _, r in fun_meta.iterrows():
            meta_f[(r["patient"], r["eye"])] = {"dme": r["dme"], "dr": r["dr"]}

    rows = []
    for key in sorted(set(f_idx) & set(o_idx)):
        patient, eye = key
        # pick first sample each (deterministic)
        f_path = sorted(f_idx[key])[0]
        o_path = sorted(o_idx[key])[0]
        mo = meta_o.get(key, {})
        mf = meta_f.get(key, {})
        dme = mo.get("dme")
        if dme is None:
            dme = mf.get("dme")
        if dme is None:
            continue
        rows.append(
            {
                "patient": patient,
                "eye": eye,
                "path_oct": str(o_path.resolve()),
                "path_fundus": str(f_path.resolve()),
                "label_dme": int(dme),
                "dr_oct": mo.get("dr"),
                "dr_fundus": mf.get("dr"),
                "source_dataset": "oefi",
            }
        )
    return pd.DataFrame(rows)


def map_dr_to_ghota(dr: str | None) -> int | None:
    if dr is None or (isinstance(dr, float) and pd.isna(dr)):
        return None
    s = str(dr).strip().upper()
    if s in {"0", "NO", "NONE", "NO DR"}:
        return 0
    if s in {"NPDR", "1", "2", "3"}:
        return 1
    if s in {"PDR", "4", "5"}:
        return 2
    if s in {"-", "NAN", "UNKNOWN", ""}:
        return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-name", default="ghota_oefi_paired.csv")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = build_pairs()
    if pairs.empty:
        print("No pairs found")
        return 1
    pairs["label_dr_ghota"] = pairs["dr_oct"].map(map_dr_to_ghota)
    # fallback fundus DR
    miss = pairs["label_dr_ghota"].isna()
    if miss.any():
        fallback = pairs.loc[miss, "dr_fundus"].map(map_dr_to_ghota)
        pairs["label_dr_ghota"] = pairs["label_dr_ghota"].astype("Float64")
        pairs.loc[miss, "label_dr_ghota"] = fallback.to_numpy()
    pairs["label_dr_ghota"] = pairs["label_dr_ghota"].astype("Float64")

    out = OUT / args.out_name
    pairs.to_csv(out, index=False)
    print(f"Wrote {out} n={len(pairs)} DME+={int(pairs['label_dme'].sum())} DME-={int((pairs['label_dme']==0).sum())}")
    print("DR GHOTA counts:", pairs["label_dr_ghota"].value_counts(dropna=False).to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
