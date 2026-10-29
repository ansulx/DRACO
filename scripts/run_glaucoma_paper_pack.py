#!/usr/bin/env python3
"""Train remaining GLAUCO paper runs, evaluate, and write comparison plots/docs.

Designed for a single RTX A4000 session (~2-4 hours). Skips H100 transfer.
"""

from __future__ import annotations

import json
import subprocess
import sys
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

OUT_FIG = ROOT / "results" / "glaucoma_figures"
OUT_JSON = ROOT / "results" / "glaucoma_paper_pack.json"
DOC = ROOT / "docs" / "GLAUCOMA_RESULTS_2026.md"
CAT = ROOT / "data" / "processed" / "glaucoma" / "catalogs"

# Literature numbers (published / reported)
LIT = {
    "HAMM tri-modal (GLEAM, 2026)": {
        "accuracy": 0.8108,
        "macro_f1": 0.7590,
        "auroc": 0.9303,
        "kappa": 0.9007,
        "source": "arXiv:2603.12800 / IEEE TMI 2026",
        "link": "https://arxiv.org/abs/2603.12800",
    },
    "M2F3 fused (GAMMA)": {
        "kappa": 0.9015,
        "note": "dual fundus+OCT on GAMMA (not GLEAM)",
        "source": "TVST 2025",
        "link": "https://doi.org/10.1167/tvst.14.11.8",
    },
    "Harvard-GDP EfficientNet RNFLT": {
        "auc": 0.90,
        "source": "Harvard-GDP paper benchmark",
        "link": "https://github.com/Harvard-Ophthalmology-AI-Lab/Harvard-GF",
    },
}

TRAIN_CONFIGS = [
    "configs/glaucoma_fusion_gleam_trimodal_xattn.yaml",
    "configs/glaucoma_fusion_gleam_trimodal_late.yaml",
    "configs/glaucoma_fusion_gleam_dual_xattn_eff.yaml",
    "configs/glaucoma_fusion_gleam_late.yaml",
    "configs/glaucoma_fusion_gleam_binary_xattn.yaml",
]

# Already-trained specialists / prior fusion to include in the report
EXISTING = [
    ("glauco_efficientnet_harvard_gf_full", "OCT/RNFLT · Harvard-GF", "auc"),
    ("glauco_retfound_cfp_airogs_light", "Fundus · AIROGS-light", "auc"),
    ("glauco_efficientnet_harvard_gd_full", "OCT/RNFLT · Harvard-GD", "auc"),
    ("glauco_fusion_gleam_crossattn", "Dual fusion (RETFound+Eff, prior)", "macro_f1"),
]


def run(cmd: list[str]) -> int:
    print("\n>>>", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def ensure_single_modality_catalogs() -> None:
    """Derive per-modality GLEAM catalogs for optional ablations / inference."""
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


def train_all() -> None:
    for cfg in TRAIN_CONFIGS:
        cfg_path = ROOT / cfg
        with open(cfg_path, encoding="utf-8") as f:
            run_name = yaml.safe_load(f)["run_name"]
        ckpt = ROOT / "checkpoints" / run_name / "best.pt"
        if ckpt.exists():
            print(f"SKIP (exists): {run_name}")
            continue
        rc = run([str(PY), "-u", "-m", "glauco.train", "--config", cfg])
        if rc != 0:
            print(f"WARN: train failed for {cfg} rc={rc}")


def evaluate_run(run_name: str, test_csv: Path) -> dict | None:
    ckpt = ROOT / "checkpoints" / run_name / "best.pt"
    if not ckpt.exists():
        return None
    out = ROOT / "checkpoints" / run_name / "eval_official_test.json"
    run(
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
    if not out.exists():
        return None
    with open(out, encoding="utf-8") as f:
        return json.load(f)


def collect_results() -> dict:
    test_csv = CAT / "gleam_paired_test.csv"
    pack: dict = {"literature": LIT, "our_runs": {}, "specialists": {}}

    for cfg in TRAIN_CONFIGS + ["configs/glaucoma_fusion_gleam_crossattn.yaml"]:
        cfg_path = ROOT / cfg
        if not cfg_path.exists():
            continue
        with open(cfg_path, encoding="utf-8") as f:
            c = yaml.safe_load(f)
        name = c["run_name"]
        report = evaluate_run(name, test_csv)
        best = ROOT / "checkpoints" / name / "best_metrics.json"
        val = None
        if best.exists():
            with open(best, encoding="utf-8") as f:
                val = json.load(f)
        entry = {"config": cfg, "val": val, "test": None}
        if report:
            # evaluate.py nests under csv stem
            for k, v in report.items():
                if isinstance(v, dict) and ("accuracy" in v or "auc" in v):
                    entry["test"] = v
                    break
        pack["our_runs"][name] = entry

    for name, label, _ in EXISTING:
        best = ROOT / "checkpoints" / name / "best_metrics.json"
        if best.exists():
            with open(best, encoding="utf-8") as f:
                pack["specialists"][name] = {"label": label, "metrics": json.load(f)}
        hold = ROOT / "checkpoints" / name / "eval_holdout_test.json"
        off = ROOT / "checkpoints" / name / "eval_official_test.json"
        for p in (hold, off):
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    pack["specialists"][name]["external"] = json.load(f)
                break

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(pack, f, indent=2)
    return pack


def make_figures(pack: dict) -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    # --- Fig 1: GLEAM 4-class Acc / F1 ours vs HAMM ---
    rows = []
    for name, entry in pack["our_runs"].items():
        t = entry.get("test") or {}
        if not t or "macro_f1" not in t:
            continue
        if "binary" in name:
            continue
        rows.append(
            {
                "run": name.replace("glauco_fusion_gleam_", ""),
                "accuracy": t.get("accuracy", 0),
                "macro_f1": t.get("macro_f1", 0),
            }
        )
    if rows:
        df = pd.DataFrame(rows).sort_values("macro_f1")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = range(len(df))
        ax.bar([i - 0.2 for i in x], df["accuracy"], width=0.4, label="Accuracy")
        ax.bar([i + 0.2 for i in x], df["macro_f1"], width=0.4, label="Macro-F1")
        ax.axhline(0.8108, color="#c44", ls="--", lw=1.5, label="HAMM Acc 81.1%")
        ax.axhline(0.7590, color="#e80", ls=":", lw=1.5, label="HAMM F1 75.9%")
        ax.set_xticks(list(x))
        ax.set_xticklabels(df["run"], rotation=25, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Score")
        ax.set_title("GLEAM 4-class: our fusion vs HAMM 2026")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_FIG / "gleam_vs_hamm.png", dpi=160)
        plt.close(fig)

    # --- Fig 2: Specialist AUC bar ---
    specs = []
    for name, info in pack.get("specialists", {}).items():
        m = info.get("metrics") or {}
        ext = info.get("external") or {}
        # dig for auc
        auc = m.get("auc")
        if auc is None and isinstance(ext, dict):
            # nested
            for v in ext.values():
                if isinstance(v, dict) and "auc" in v:
                    auc = v["auc"]
                    break
            if auc is None:
                auc = ext.get("auc")
        if auc is not None:
            specs.append({"label": info.get("label", name), "auc": auc})
    if specs:
        df = pd.DataFrame(specs)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(df["label"], df["auc"], color="#2a6f97")
        ax.axvline(0.90, color="#c44", ls="--", label="Harvard RNFLT bar 0.90")
        ax.set_xlim(0.5, 1.0)
        ax.set_xlabel("AUC")
        ax.set_title("Single-modality specialists (this project)")
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_FIG / "specialist_auc.png", dpi=160)
        plt.close(fig)

    # --- Fig 3: training curves for best trimodal if present ---
    hist = ROOT / "checkpoints" / "glauco_fusion_gleam_trimodal_xattn" / "history.json"
    if hist.exists():
        with open(hist, encoding="utf-8") as f:
            history = json.load(f)
        epochs = [h["epoch"] for h in history]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, [h["macro_f1"] for h in history], label="val macro-F1")
        ax.plot(epochs, [h["accuracy"] for h in history], label="val accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title("Tri-modal cross-attn training (GLEAM val)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_FIG / "trimodal_training_curve.png", dpi=160)
        plt.close(fig)


def write_doc(pack: dict) -> None:
    lines = [
        "# GLAUCO — Full Results Pack (A4000)",
        "",
        "**Hardware:** NVIDIA RTX A4000 16 GB (no H100 required).",
        "**Goal:** Dataset inventory → 2026 literature comparison → trained models → graphs.",
        "",
        "## 1. Datasets we used",
        "",
        "| Dataset | Modality | N | Role | Link |",
        "|---------|----------|---|------|------|",
        "| Harvard-GD | RNFLT | 500 | OCT smoke / high AUC | [HF Harvard-GD](https://huggingface.co/datasets/harvardairobotics/Harvard-GD) |",
        "| Harvard-GDP | RNFLT | 1,000 | OCT scale | [HF Harvard-GDP](https://huggingface.co/datasets/harvardairobotics/Harvard-GDP) |",
        "| Harvard-GF | RNFLT (+ OCT) | 3,300 | Primary OCT benchmark | [Zenodo](https://zenodo.org/records/13178701) |",
        "| AIROGS-light-v2 | Fundus | ~9.5k | Primary fundus | [Kaggle](https://www.kaggle.com/datasets/deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2) |",
        "| GLEAM | SLO + cpOCT + VF | 1,200 | Fusion / staging (2026) | [Kaggle](https://www.kaggle.com/datasets/zhangyiyinge/gleam-dataset) |",
        "",
        "Optional external (not required for this pack): [REFUGE](https://ieee-dataport.org/documents/refuge-retinal-fundus-glaucoma-challenge), [GAMMA](https://gamma.grand-challenge.org/).",
        "",
        "## 2. 2026 / recent papers we compare against",
        "",
        "| Paper | What to match | Link |",
        "|-------|---------------|------|",
        "| **HAMM / GLEAM (TMI 2026)** | Tri-modal Acc≈81%, F1≈76%, AUROC≈93% | [arXiv:2603.12800](https://arxiv.org/abs/2603.12800) · [GitHub](https://github.com/microewing/HAMM) |",
        "| **M2F3 (TVST)** | Dual fusion κ≈0.90 on GAMMA | [DOI](https://doi.org/10.1167/tvst.14.11.8) |",
        "| **RETFound (Nature 2023)** | Fundus/OCT foundation backbone | [Nature](https://www.nature.com/articles/s41586-023-06555-x) |",
        "| Harvard RNFLT EfficientNet | OCT AUC ≈ 0.90 | [Harvard-GF](https://github.com/Harvard-Ophthalmology-AI-Lab/Harvard-GF) |",
        "",
        "## 3. Our specialist results (screening)",
        "",
        "| Model | Dataset | Metric |",
        "|-------|---------|--------|",
    ]
    for name, info in pack.get("specialists", {}).items():
        m = info.get("metrics") or {}
        bits = []
        if m.get("auc") is not None:
            bits.append(f"AUC {m['auc']:.3f}")
        if m.get("macro_f1") is not None:
            bits.append(f"F1 {m['macro_f1']:.3f}")
        lines.append(f"| {info.get('label', name)} | `{name}` | {', '.join(bits)} |")

    lines += [
        "",
        "## 4. GLEAM fusion results (test set)",
        "",
        "| Run | Acc | Macro-F1 | Notes |",
        "|-----|-----|----------|-------|",
    ]
    for name, entry in pack.get("our_runs", {}).items():
        t = entry.get("test") or {}
        if not t:
            lines.append(f"| `{name}` | — | — | no test eval |")
            continue
        acc = t.get("accuracy")
        f1 = t.get("macro_f1")
        br = t.get("binary_remapped") or {}
        note = ""
        if br.get("auc") is not None:
            note = f"binary AUC {br['auc']:.3f}"
        if t.get("auc") is not None:
            note = f"AUC {t['auc']:.3f}"
        acc_s = f"{acc:.3f}" if acc is not None else "—"
        f1_s = f"{f1:.3f}" if f1 is not None else "—"
        lines.append(f"| `{name}` | {acc_s} | {f1_s} | {note} |")

    lines += [
        "",
        f"**Literature bar (HAMM tri-modal):** Acc {LIT['HAMM tri-modal (GLEAM, 2026)']['accuracy']:.3f}, "
        f"F1 {LIT['HAMM tri-modal (GLEAM, 2026)']['macro_f1']:.3f}, "
        f"AUROC {LIT['HAMM tri-modal (GLEAM, 2026)']['auroc']:.3f}.",
        "",
        "## 5. Figures",
        "",
        "- `results/glaucoma_figures/gleam_vs_hamm.png`",
        "- `results/glaucoma_figures/specialist_auc.png`",
        "- `results/glaucoma_figures/trimodal_training_curve.png`",
        "",
        "## 6. How to demo / retrain",
        "",
        "```powershell",
        "cd C:\\Users\\HP\\OneDrive\\Desktop\\OCT",
        ".\\.venv\\Scripts\\python.exe scripts\\run_glaucoma_paper_pack.py",
        ".\\.venv\\Scripts\\python.exe scripts\\glauco_infer.py --help",
        "```",
        "",
        "Machine-readable pack: `results/glaucoma_paper_pack.json`.",
        "",
    ]
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {DOC}")


def main() -> int:
    print(f"Using Python: {PY}")
    # Rebuild GLEAM catalogs if missing
    if not (CAT / "gleam_paired_train.csv").exists():
        run([str(PY), "scripts/build_glaucoma_catalogs.py", "--dataset", "gleam"])
    ensure_single_modality_catalogs()
    train_all()
    pack = collect_results()
    make_figures(pack)
    write_doc(pack)
    print(f"\nDone. Pack: {OUT_JSON}")
    print(f"Figures: {OUT_FIG}")
    print(f"Doc: {DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
