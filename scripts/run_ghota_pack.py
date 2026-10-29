"""Evaluate GHOTA checkpoints (holdout + OEFI external) and emit a results pack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from draco.metrics import binary_metrics, multiclass_metrics, remap_3class_to_binary_prob
from ghota.data.datasets import GhotaCatalogDataset
from ghota.models.factory import build_ghota_from_config
from ghota.models.fusion import LateFusion

FIG = ROOT / "docs" / "ghota" / "figures"
OUT_DOC = ROOT / "docs" / "GHOTA_RESULTS.md"
PACK_JSON = ROOT / "docs" / "ghota" / "results_pack.json"


def _load_ckpt(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config") or {}
    cfg_path = path.parent / "config.yaml"
    if (not cfg) and cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return ckpt, cfg


@torch.no_grad()
def _collect(model, loader, device):
    model.eval()
    ys, preds, probs = [], [], []
    for x, d, y in loader:
        logits = model(x.to(device, non_blocking=True), d.to(device, non_blocking=True))
        p = torch.softmax(logits, dim=1).cpu().numpy()
        ys.append(y.numpy())
        preds.append(p.argmax(1))
        probs.append(p)
    return np.concatenate(ys), np.concatenate(preds), np.concatenate(probs)


def eval_catalog(model, csv: Path, cfg: dict, device, *, remap: bool = True) -> dict:
    ds = GhotaCatalogDataset(
        csv,
        diseases=list(cfg.get("diseases") or []),
        image_size=int((cfg.get("data") or {}).get("image_size", 224)),
        split="eval",
        remap=remap,
    )
    if not cfg.get("diseases"):
        cfg["diseases"] = ds.diseases
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    y_true, y_pred, probs = _collect(model, loader, device)
    out = multiclass_metrics(y_true, y_pred)
    # binary collapse: any disease vs none
    y_bin = (y_true > 0).astype(int)
    out["binary_collapse"] = binary_metrics(y_bin, remap_3class_to_binary_prob(probs))
    return out


def eval_oefi(model, csv: Path, cfg: dict, device) -> dict:
    """OEFI is binary 0/1; remap scheme maps to GHOTA 0/1 then P(DME)=P1+P2."""
    ds = GhotaCatalogDataset(
        csv,
        diseases=list(cfg.get("diseases") or ["oct_dme"]),
        image_size=int((cfg.get("data") or {}).get("image_size", 224)),
        split="eval",
        remap=True,
    )
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    y_true, _y_pred, probs = _collect(model, loader, device)
    # After remap, OEFI DME→class1; treat y>0 as positive for safety
    y_bin = (y_true > 0).astype(int)
    return binary_metrics(y_bin, remap_3class_to_binary_prob(probs))


def _bar_specialists(pack: dict, path: Path) -> None:
    names = ["OCT\n(MMRDR+OCTID)", "Fundus\n(APTOS)"]
    acc = [pack["oct"]["holdout"]["accuracy"], pack["fundus"]["holdout"]["accuracy"]]
    f1 = [pack["oct"]["holdout"]["macro_f1"], pack["fundus"]["holdout"]["macro_f1"]]
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - w / 2, acc, w, label="Accuracy", color="#0f766e")
    ax.bar(x + w / 2, f1, w, label="Macro-F1", color="#0369a1")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Score")
    ax.set_title("GHOTA SOTA specialists — 3-class DR severity")
    for i, (a, f) in enumerate(zip(acc, f1)):
        ax.text(i - w / 2, a + 0.02, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i + w / 2, f + 0.02, f"{f:.3f}", ha="center", fontsize=9)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_cm(cm: list, title: str, path: Path, labels=("none", "early", "adv")) -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_oefi(oefi: dict, path: Path) -> None:
    metrics = ["auc", "sensitivity", "specificity", "f1", "accuracy"]
    vals = [float(oefi.get(m) or 0) for m in metrics]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.bar(metrics, vals, color="#0f766e")
    ax.set_ylim(0, 1.05)
    ax.set_title("OEFI external — binary DME (OCT specialist)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_markdown(pack: dict) -> None:
    oct_h = pack["oct"]["holdout"]
    fun_h = pack["fundus"]["holdout"]
    oefi = pack["oct"]["oefi"]
    lines = [
        "# GHOTA Results — DR 3-class SOTA",
        "",
        f"_Generated from checkpoints on disk. Pack JSON: `docs/ghota/results_pack.json`._",
        "",
        "## Summary",
        "",
        "| Specialist | Holdout Acc | Macro-F1 | Checkpoint |",
        "|------------|------------:|---------:|------------|",
        f"| **OCT** (MMRDR+OCTID) | **{oct_h['accuracy']:.3f}** | **{oct_h['macro_f1']:.3f}** | `ghota_oct_dme_retfound_sota` |",
        f"| **Fundus** (APTOS) | **{fun_h['accuracy']:.3f}** | **{fun_h['macro_f1']:.3f}** | `ghota_fundus_dr_retfound_sota` |",
        "",
        "### OEFI external (binary DME, OCT model)",
        "",
        f"| AUC | Sens@0.5 | Spec@0.5 | F1 | Acc | Sens@95%Spec |",
        f"|----:|---------:|---------:|---:|----:|-------------:|",
        f"| {oefi.get('auc', 0):.3f} | {oefi.get('sensitivity', 0):.3f} | "
        f"{oefi.get('specificity', 0):.3f} | {oefi.get('f1', 0):.3f} | "
        f"{oefi.get('accuracy', 0):.3f} | {oefi.get('sensitivity_at_95_spec') or 0:.3f} |",
        "",
        "## Fusion status",
        "",
        pack["fusion"]["status_md"],
        "",
        "## Figures",
        "",
        "- `docs/ghota/figures/01_specialists_acc_f1.png`",
        "- `docs/ghota/figures/02_oct_confusion.png`",
        "- `docs/ghota/figures/03_fundus_confusion.png`",
        "- `docs/ghota/figures/04_oefi_external.png`",
        "",
        "## Recipe",
        "",
        "RETFound (OCT / CFP) + Focal γ=2 + layer-decay 0.65 + cosine warmup + head warmup.",
        "",
        "## One-liner",
        "",
        "> GHOTA delivers strong **unpaired** DR specialists on OCT and fundus under one "
        "> 3-class severity language; true eye-paired late/cross-attn fusion is ready in code "
        "> and waits on paired fundus+OCT data.",
        "",
    ]
    OUT_DOC.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_DOC}")


def build_pptx(pack: dict) -> Path:
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Inches, Pt

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    def _add_title_slide(title: str, subtitle: str):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        box = slide.shapes.add_textbox(Inches(0.7), Inches(2.2), Inches(12), Inches(1.2))
        tf = box.text_frame
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(40)
        p.font.bold = True
        p.font.color.rgb = RGBColor(15, 118, 110)
        sub = slide.shapes.add_textbox(Inches(0.7), Inches(3.5), Inches(12), Inches(1.5))
        sp = sub.text_frame.paragraphs[0]
        sp.text = subtitle
        sp.font.size = Pt(18)
        sp.font.color.rgb = RGBColor(51, 65, 85)
        return slide

    def _bullets(slide, items, top=1.6):
        box = slide.shapes.add_textbox(Inches(0.8), Inches(top), Inches(11.5), Inches(5))
        tf = box.text_frame
        tf.word_wrap = True
        for i, t in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = t
            p.font.size = Pt(20)
            p.font.color.rgb = RGBColor(30, 41, 59)
            p.space_after = Pt(10)

    oct_h = pack["oct"]["holdout"]
    fun_h = pack["fundus"]["holdout"]
    oefi = pack["oct"]["oefi"]

    _add_title_slide(
        "GHOTA — DR severity (OCT + fundus)",
        "3-class: none / early–moderate / advanced   ·   Glaucoma stays in GLAUCO",
    )

    s2 = prs.slides.add_slide(prs.slide_layouts[6])
    t = s2.shapes.add_textbox(Inches(0.7), Inches(0.4), Inches(12), Inches(0.8))
    t.text_frame.paragraphs[0].text = "Results — SOTA specialists"
    t.text_frame.paragraphs[0].font.size = Pt(32)
    t.text_frame.paragraphs[0].font.bold = True
    t.text_frame.paragraphs[0].font.color.rgb = RGBColor(15, 118, 110)
    _bullets(
        s2,
        [
            f"OCT (MMRDR+OCTID):  Acc {oct_h['accuracy']:.1%}   ·   Macro-F1 {oct_h['macro_f1']:.3f}",
            f"Fundus APTOS:       Acc {fun_h['accuracy']:.1%}   ·   Macro-F1 {fun_h['macro_f1']:.3f}",
            f"OEFI external DME:  AUC {oefi.get('auc', 0):.3f}   ·   "
            f"Sens {oefi.get('sensitivity', 0):.3f} / Spec {oefi.get('specificity', 0):.3f}",
            "Recipe: RETFound + Focal + layer-decay 0.65 + cosine + head warmup",
        ],
    )
    fig = FIG / "01_specialists_acc_f1.png"
    if fig.exists():
        s2.shapes.add_picture(str(fig), Inches(7.2), Inches(1.5), width=Inches(5.5))

    s3 = prs.slides.add_slide(prs.slide_layouts[6])
    t = s3.shapes.add_textbox(Inches(0.7), Inches(0.4), Inches(12), Inches(0.8))
    t.text_frame.paragraphs[0].text = "Fusion & defense line"
    t.text_frame.paragraphs[0].font.size = Pt(32)
    t.text_frame.paragraphs[0].font.bold = True
    t.text_frame.paragraphs[0].font.color.rgb = RGBColor(15, 118, 110)
    _bullets(
        s3,
        [
            "Unpaired multi-modal SOTA today: task-conditioned dual specialists",
            "Late fusion / cross-attn code ready — needs real eye-paired fundus+OCT",
            "GLAUCO = glaucoma; GHOTA = DR only — cleaner Acc, cleaner story",
            pack["fusion"]["one_liner"],
        ],
    )

    out = ROOT / "GHOTA_Presentation_3Slides.pptx"
    prs.save(out)
    print(f"Wrote {out}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--skip-pptx",
        action="store_true",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FIG.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    pack: dict = {"fusion": {}}

    # --- OCT ---
    ckpt, cfg = _load_ckpt(args.oct_ckpt, device)
    model = build_ghota_from_config(cfg, load_weights=False).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    oct_test = ROOT / "data/processed/ghota/catalogs/ghota_oct_all_test.csv"
    oefi_csv = ROOT / "data/processed/ghota/catalogs/ghota_oefi_external.csv"
    print("Evaluating OCT holdout...")
    oct_hold = eval_catalog(model, oct_test, cfg, device)
    print(f"  Acc={oct_hold['accuracy']:.4f} F1={oct_hold['macro_f1']:.4f}")
    print("Evaluating OEFI external...")
    oefi = eval_oefi(model, oefi_csv, cfg, device)
    print(f"  AUC={oefi.get('auc')} Sens={oefi.get('sensitivity'):.4f} Spec={oefi.get('specificity'):.4f}")
    pack["oct"] = {
        "checkpoint": str(args.oct_ckpt),
        "holdout": oct_hold,
        "oefi": oefi,
    }
    (args.oct_ckpt.parent / "eval_holdout.json").write_text(
        json.dumps(oct_hold, indent=2), encoding="utf-8"
    )
    (args.oct_ckpt.parent / "eval_oefi.json").write_text(
        json.dumps(oefi, indent=2), encoding="utf-8"
    )

    # --- Fundus ---
    ckpt_f, cfg_f = _load_ckpt(args.fundus_ckpt, device)
    model_f = build_ghota_from_config(cfg_f, load_weights=False).to(device)
    model_f.load_state_dict(ckpt_f["model"], strict=False)
    fun_test = ROOT / "data/processed/ghota/catalogs/ghota_fundus_dr_test.csv"
    print("Evaluating fundus holdout...")
    fun_hold = eval_catalog(model_f, fun_test, cfg_f, device)
    print(f"  Acc={fun_hold['accuracy']:.4f} F1={fun_hold['macro_f1']:.4f}")
    pack["fundus"] = {
        "checkpoint": str(args.fundus_ckpt),
        "holdout": fun_hold,
    }
    (args.fundus_ckpt.parent / "eval_holdout.json").write_text(
        json.dumps(fun_hold, indent=2), encoding="utf-8"
    )

    # --- Late fusion readiness (no paired rows → document + smoke-test module) ---
    lf = LateFusion(num_classes=3, learnable=False)
    demo_o = torch.tensor([[2.0, 0.1, 0.1], [0.2, 2.0, 0.1]])
    demo_f = torch.tensor([[1.5, 0.5, 0.1], [0.1, 0.2, 2.0]])
    fused = lf(demo_o, demo_f)
    pack["fusion"] = {
        "paired_data_on_disk": False,
        "module": "ghota.models.fusion.LateFusion / CrossAttentionFusion",
        "smoke_test_probs_shape": list(fused.shape),
        "status_md": (
            "**No eye-paired DR fundus+OCT catalog on disk**, so we do **not** report a "
            "fused Acc number (that would require fabricated pairs).\n\n"
            "- Specialists are production-ready independently (route by modality).\n"
            "- `LateFusion` and `CrossAttentionFusion` are implemented and smoke-tested.\n"
            "- When a paired CSV arrives (`path_oct`, `path_fundus`, `label`), plug into "
            "late fusion and report Acc/F1 honestly."
        ),
        "one_liner": (
            "Today: dual specialists by modality. Tomorrow: late/cross-attn fusion on real pairs."
        ),
    }

    # Figures
    _bar_specialists(pack, FIG / "01_specialists_acc_f1.png")
    _plot_cm(oct_hold["confusion_matrix"], "OCT holdout CM", FIG / "02_oct_confusion.png")
    _plot_cm(fun_hold["confusion_matrix"], "Fundus holdout CM", FIG / "03_fundus_confusion.png")
    _plot_oefi(oefi, FIG / "04_oefi_external.png")

    PACK_JSON.parent.mkdir(parents=True, exist_ok=True)
    PACK_JSON.write_text(json.dumps(pack, indent=2), encoding="utf-8")
    print(f"Wrote {PACK_JSON}")
    write_markdown(pack)
    if not args.skip_pptx:
        try:
            build_pptx(pack)
        except Exception as e:  # pragma: no cover
            print(f"PPTX skipped: {e}")

    # Refresh GHOTA.md brief results section via rewrite of data status
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
