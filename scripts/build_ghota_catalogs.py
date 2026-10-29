"""Build GHOTA DR-only catalogs (OCT DME + optional fundus DR). Glaucoma excluded."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "ghota" / "catalogs"


def _write(df: pd.DataFrame, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    df.to_csv(path, index=False)
    print(f"Wrote {path} ({len(df)} rows)  classes={df['label'].value_counts().to_dict()}")
    return path


def from_mmrdr() -> None:
    """MMRDR OCT DME — already 0/1/2."""
    for split, out_name in [("train", "ghota_oct_dme_train.csv"), ("test", "ghota_oct_dme_test.csv")]:
        src = None
        for base in [
            ROOT / "data" / "processed" / "mmrdr" / "catalogs" / f"mmrdr_{split}.csv",
            ROOT / "data" / "processed" / "catalogs" / f"mmrdr_{split}.csv",
        ]:
            if base.exists():
                src = base
                break
        if src is None:
            print(f"Skip MMRDR {split}: catalog not found")
            continue
        df = pd.read_csv(src)
        path_col = "path" if "path" in df.columns else df.columns[0]
        label_col = "label" if "label" in df.columns else "dme"
        out = pd.DataFrame(
            {
                "path": df[path_col],
                "label": df[label_col].astype(int),
                "disease": "oct_dme",  # task id (legacy column name)
                "scheme": "oct_dme",
                "modality": "oct",
                "split": split,
                "source_dataset": "mmrdr",
            }
        )
        _write(out, out_name)


def from_octid() -> None:
    """OCTID Normal/DR supplement → soft 3-class (DR→early_mod)."""
    src = ROOT / "data" / "processed" / "catalogs" / "octid_supplement.csv"
    if not src.exists():
        print("Skip OCTID: catalog not found")
        return
    df = pd.read_csv(src)
    # Stratified holdout so supplement isn't train-only leakage on Acc claims
    from sklearn.model_selection import train_test_split

    tr, te = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    for split_df, split, out_name in [
        (tr, "train", "ghota_octid_train.csv"),
        (te, "test", "ghota_octid_test.csv"),
    ]:
        out = pd.DataFrame(
            {
                "path": split_df["path"],
                "label": split_df["label"].astype(int),
                "disease": "oct_dme",
                "scheme": "octid_binary",
                "modality": "oct",
                "split": split,
                "source_dataset": "octid",
            }
        )
        _write(out, out_name)


def from_oefi_external() -> None:
    """OEFI binary DME — external set only (do not mix into Acc training)."""
    src = ROOT / "data" / "processed" / "catalogs" / "oefi_external.csv"
    if not src.exists():
        print("Skip OEFI: catalog not found")
        return
    df = pd.read_csv(src)
    out = pd.DataFrame(
        {
            "path": df["path"],
            "label": df["label"].astype(int),
            "disease": "oct_dme",
            "scheme": "oefi_binary",
            "modality": "oct",
            "split": "external",
            "source_dataset": "oefi",
        }
    )
    _write(out, "ghota_oefi_external.csv")


def from_idrid() -> None:
    """IDRiD fundus ICDR 0–4 — same modality as APTOS / GHOTA fundus."""
    root = ROOT / "data" / "raw" / "idrid" / "disease_grading"
    if not root.exists():
        print("Skip IDRiD: run scripts/download_ghota_same_modality.py --dataset idrid")
        return
    # Find grading CSV(s)
    csvs = list(root.rglob("*.csv"))
    if not csvs:
        print(f"Skip IDRiD: no CSV under {root}")
        return
    # Prefer files mentioning retinopathy / grading
    meta_path = next(
        (
            p
            for p in csvs
            if "retinopathy" in p.name.lower()
            or "grading" in p.name.lower()
            or "dr" in p.name.lower()
        ),
        csvs[0],
    )
    meta = pd.read_csv(meta_path)
    cols = {c.lower().strip(): c for c in meta.columns}
    # common: Image name, Retinopathy grade
    name_col = cols.get("image name") or cols.get("image") or cols.get("id") or meta.columns[0]
    grade_col = (
        cols.get("retinopathy grade")
        or cols.get("retinopathy_grade")
        or cols.get("diagnosis")
        or cols.get("dr")
        or meta.columns[1]
    )
    # image dirs
    img_dirs = [p for p in root.rglob("*") if p.is_dir() and any(p.glob("*.jpg"))]
    if not img_dirs:
        img_dirs = [root]

    rows = []
    for _, row in meta.iterrows():
        stem = str(row[name_col]).replace(".jpg", "").replace(".JPG", "")
        path = None
        for d in img_dirs:
            for ext in (".jpg", ".JPG", ".jpeg", ".png"):
                cand = d / f"{stem}{ext}"
                if cand.exists():
                    path = cand
                    break
                # sometimes nested
                hits = list(d.rglob(f"{stem}{ext}"))
                if hits:
                    path = hits[0]
                    break
            if path:
                break
        if path is None:
            continue
        rows.append(
            {
                "path": str(path.resolve()),
                "label": int(row[grade_col]),
                "disease": "fundus_dr",
                "scheme": "fundus_dr",
                "modality": "fundus",
                "split": "train",  # reassigned below
                "source_dataset": "idrid",
            }
        )
    if not rows:
        print(f"IDRiD: CSV={meta_path.name} but no images matched")
        return
    df = pd.DataFrame(rows)
    from sklearn.model_selection import train_test_split

    tr, te = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    tr = tr.copy()
    te = te.copy()
    tr["split"] = "train"
    te["split"] = "test"
    _write(tr, "ghota_idrid_train.csv")
    _write(te, "ghota_idrid_test.csv")


def from_aptos_placeholder() -> None:
    """Wire APTOS when downloaded to data/raw/aptos or data/raw/dr/aptos."""
    candidates = [
        ROOT / "data" / "raw" / "aptos",
        ROOT / "data" / "raw" / "dr" / "aptos",
        ROOT / "aptos2019-blindness-detection",
    ]
    root = next((p for p in candidates if p.exists()), None)
    if root is None:
        print(
            "Fundus DR: APTOS not on disk yet.\n"
            "  Download Kaggle aptos2019-blindness-detection -> data/raw/aptos/\n"
            "  Then re-run: python scripts/build_ghota_catalogs.py --source fundus"
        )
        return

    # Expect train.csv with image + diagnosis (0-4)
    train_csv = root / "train.csv"
    if not train_csv.exists():
        print(f"Found {root} but no train.csv — adjust layout")
        return
    meta = pd.read_csv(train_csv)
    id_col = "id_code" if "id_code" in meta.columns else meta.columns[0]
    lab_col = "diagnosis" if "diagnosis" in meta.columns else meta.columns[1]
    rows = []
    img_dir = root / "train_images"
    if not img_dir.exists():
        img_dir = root
    for _, row in meta.iterrows():
        stem = str(row[id_col])
        path = None
        for ext in (".png", ".jpg", ".jpeg"):
            cand = img_dir / f"{stem}{ext}"
            if cand.exists():
                path = cand
                break
        if path is None:
            continue
        rows.append(
            {
                "path": str(path.resolve()),
                "label": int(row[lab_col]),  # raw ICDR 0-4; map at train time
                "disease": "fundus_dr",
                "scheme": "fundus_dr",
                "modality": "fundus",
                "split": "train",
                "source_dataset": "aptos2019",
            }
        )
    if not rows:
        print("APTOS: no images matched")
        return
    df = pd.DataFrame(rows)
    # simple holdout
    from sklearn.model_selection import train_test_split

    tr, te = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    tr = tr.copy()
    te = te.copy()
    tr["split"] = "train"
    te["split"] = "test"
    _write(tr, "ghota_fundus_dr_train.csv")
    _write(te, "ghota_fundus_dr_test.csv")


def merge_joint_dr() -> None:
    """Joint DR catalogs only (no glaucoma)."""
    trains, tests = [], []
    for p in sorted(OUT.glob("ghota_*.csv")):
        if "joint" in p.name or "glaucoma" in p.name:
            continue
        if p.name.endswith("_train.csv"):
            trains.append(pd.read_csv(p))
        elif p.name.endswith("_test.csv"):
            tests.append(pd.read_csv(p))
    # remove any leftover glaucoma files from older scaffold
    for p in OUT.glob("ghota_glaucoma*"):
        p.unlink()
        print(f"Removed legacy {p.name}")
    if trains:
        _write(pd.concat(trains, ignore_index=True), "ghota_dr_joint_train.csv")
    if tests:
        _write(pd.concat(tests, ignore_index=True), "ghota_dr_joint_test.csv")
    for legacy in ("ghota_joint_train.csv", "ghota_joint_test.csv", "ghota_dme_train.csv", "ghota_dme_test.csv"):
        lp = OUT / legacy
        # keep dme renamed copies only via oct_dme_*; delete old joint with glaucoma
        if legacy.startswith("ghota_joint") and lp.exists():
            lp.unlink()
            print(f"Removed legacy {legacy}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build GHOTA DR-only catalogs")
    parser.add_argument(
        "--source",
        default="all",
        choices=["all", "oct", "fundus", "joint"],
    )
    args = parser.parse_args()
    if args.source in ("all", "oct"):
        from_mmrdr()
        from_octid()
        from_oefi_external()
    if args.source in ("all", "fundus"):
        from_aptos_placeholder()
        from_idrid()
    if args.source in ("all", "joint"):
        merge_joint_dr()
        # Primary OCT Acc catalog = MMRDR + OCTID (exclude OEFI external)
        trains, tests = [], []
        for name in (
            "ghota_oct_dme_train.csv",
            "ghota_octid_train.csv",
        ):
            p = OUT / name
            if p.exists():
                trains.append(pd.read_csv(p))
        for name in (
            "ghota_oct_dme_test.csv",
            "ghota_octid_test.csv",
        ):
            p = OUT / name
            if p.exists():
                tests.append(pd.read_csv(p))
        if trains:
            _write(pd.concat(trains, ignore_index=True), "ghota_oct_all_train.csv")
        if tests:
            _write(pd.concat(tests, ignore_index=True), "ghota_oct_all_test.csv")
        # Fundus Acc catalog = APTOS + IDRiD
        ftr, fte = [], []
        for name in ("ghota_fundus_dr_train.csv", "ghota_idrid_train.csv"):
            p = OUT / name
            if p.exists():
                ftr.append(pd.read_csv(p))
        for name in ("ghota_fundus_dr_test.csv", "ghota_idrid_test.csv"):
            p = OUT / name
            if p.exists():
                fte.append(pd.read_csv(p))
        if ftr:
            _write(pd.concat(ftr, ignore_index=True), "ghota_fundus_all_train.csv")
        if fte:
            _write(pd.concat(fte, ignore_index=True), "ghota_fundus_all_test.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
