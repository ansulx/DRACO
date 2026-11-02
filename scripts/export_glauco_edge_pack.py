#!/usr/bin/env python3
"""Export a GLAUCO peak checkpoint as an edge-ready .pth + model card.

Focus: glaucoma only. Deployment teams consume the pack; we own peak Acc +
transparent params / memory / screening metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from glauco.evaluate import (  # noqa: E402
    collect_outputs,
    glaucoma_binary_metrics,
    load_checkpoint,
    _make_eval_loader,
)
from glauco.models import build_from_config  # noqa: E402

CLASS_NAMES_BINARY = ["normal", "glaucoma"]


def _count_params(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def _state_dict_bytes(state: dict) -> int:
    n = 0
    for v in state.values():
        if torch.is_tensor(v):
            n += v.numel() * v.element_size()
    return int(n)


def _peak_cuda_mb(model: torch.nn.Module, image_size: int, device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    model.eval()
    x = torch.randn(1, 3, image_size, image_size, device=device)
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()
    return float(torch.cuda.max_memory_allocated(device) / (1024**2))


def _architecture_name(cfg: dict) -> str:
    m = cfg.get("model") or {}
    name = str(m.get("name") or cfg.get("modality") or "unknown")
    return name


def build_model_stats(model: torch.nn.Module, cfg: dict, device: torch.device) -> dict[str, Any]:
    image_size = int((cfg.get("data") or {}).get("image_size") or cfg.get("image_size") or 224)
    params = _count_params(model)
    sd = model.state_dict()
    weight_bytes = _state_dict_bytes(sd)
    peak_mb = _peak_cuda_mb(model, image_size, device)
    return {
        "architecture": _architecture_name(cfg),
        "modality": cfg.get("modality", "fundus"),
        "num_classes": int(cfg.get("num_classes", 2)),
        "class_names": CLASS_NAMES_BINARY if int(cfg.get("num_classes", 2)) == 2 else None,
        "input": {
            "channels": 3,
            "height": image_size,
            "width": image_size,
            "layout": "NCHW",
            "dtype": "float32",
        },
        "parameters": params,
        "parameters_millions": round(params["total"] / 1e6, 3),
        "weights_bytes": weight_bytes,
        "weights_mb": round(weight_bytes / (1024**2), 3),
        "inference_peak_cuda_mb": None if peak_mb is None else round(peak_mb, 2),
        "device_profiled": str(device),
    }


def enrich_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize naming: sensitivity≡recall for binary positive class."""
    out = dict(raw)
    # Explicit aliases for supervisor / edge checklist
    if "sensitivity" in out and "recall" not in out:
        out["recall"] = out["sensitivity"]
    if "recall" in out and "sensitivity" not in out:
        out["sensitivity"] = out["recall"]
    if "ppv" in out and "precision" not in out:
        out["precision"] = out["ppv"]
    # Ordered pack
    ordered_keys = [
        "n",
        "accuracy",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "f1",
        "auc",
        "average_precision",
        "ppv",
        "npv",
        "brier",
        "threshold",
        "sensitivity_at_85_spec",
        "sensitivity_at_90_spec",
        "sensitivity_at_95_spec",
        "threshold_at_85_spec",
        "threshold_at_90_spec",
        "threshold_at_95_spec",
        "confusion_matrix",
        "positives",
        "negatives",
    ]
    pack = {k: out[k] for k in ordered_keys if k in out}
    pack["all"] = out
    return pack


def eval_csv(model, cfg, csv: Path, args, device) -> dict[str, Any]:
    binary = bool(cfg.get("binary", True))
    loader = _make_eval_loader(cfg, csv, args, binary)
    y_true, probs = collect_outputs(model, loader, device)
    if int(cfg.get("num_classes", 2)) == 2:
        raw = glaucoma_binary_metrics(y_true, probs[:, 1], threshold=args.threshold)
    else:
        raise SystemExit("Edge pack currently supports binary glaucoma only.")
    return enrich_metrics(raw)


def export_pack(
    checkpoint: Path,
    out_dir: Path,
    test_csvs: list[Path],
    *,
    threshold: float,
    batch_size: int,
    device: torch.device,
    tag: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt, cfg = load_checkpoint(checkpoint, device)
    num_classes = 2 if bool(cfg.get("binary", True)) else int(cfg.get("num_classes", 2))
    model = build_from_config(cfg, num_classes=num_classes, load_weights=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    stats = build_model_stats(model, cfg, device)

    class Args:
        pass

    args = Args()
    args.batch_size = batch_size
    args.image_size = int((cfg.get("data") or {}).get("image_size") or 224)
    args.num_workers = 2
    args.modality = cfg.get("modality", "fundus")
    args.threshold = threshold

    metrics_by_split: dict[str, Any] = {}
    for csv in test_csvs:
        csv = Path(csv)
        if not csv.is_absolute():
            csv = ROOT / csv
        if not csv.exists():
            print(f"skip missing catalog {csv}")
            continue
        # Probe a few paths — skip split if images were moved off-disk
        import pandas as pd

        sample = pd.read_csv(csv).head(8)
        if "path" in sample.columns:
            ok = sum(Path(str(p)).exists() for p in sample["path"])
            if ok == 0:
                print(f"skip unavailable images for {csv.name}")
                continue
        print(f"eval {csv.name} ...")
        metrics_by_split[csv.stem] = eval_csv(model, cfg, csv, args, device)

    primary = next(iter(metrics_by_split.values()), {})
    primary_flat = primary.get("all", primary) if isinstance(primary, dict) else {}

    payload = {
        "format_version": "glauco_edge_v1",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "task": "glaucoma_binary_screening",
        "product": "GLAUCO",
        "source_checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
        "run_name": cfg.get("run_name") or checkpoint.parent.name,
        "architecture": stats["architecture"],
        "modality": stats["modality"],
        "num_classes": stats["num_classes"],
        "class_names": stats["class_names"] or CLASS_NAMES_BINARY,
        "class_index": {0: "normal", 1: "glaucoma"},
        "input": stats["input"],
        "threshold": float(threshold),
        "model_stats": {
            "parameters_total": stats["parameters"]["total"],
            "parameters_trainable": stats["parameters"]["trainable"],
            "parameters_millions": stats["parameters_millions"],
            "weights_mb": stats["weights_mb"],
            "weights_bytes": stats["weights_bytes"],
            "inference_peak_cuda_mb": stats["inference_peak_cuda_mb"],
        },
        "metrics_primary": {
            "accuracy": primary_flat.get("accuracy"),
            "precision": primary_flat.get("precision") or primary_flat.get("ppv"),
            "recall": primary_flat.get("recall") or primary_flat.get("sensitivity"),
            "sensitivity": primary_flat.get("sensitivity"),
            "specificity": primary_flat.get("specificity"),
            "f1": primary_flat.get("f1"),
            "auc": primary_flat.get("auc"),
            "sensitivity_at_95_spec": primary_flat.get("sensitivity_at_95_spec"),
        },
        "metrics_by_split": metrics_by_split,
        "train_metrics_at_best": ckpt.get("metrics"),
        "config": cfg,
        "state_dict": model.state_dict(),
    }

    pth_path = out_dir / f"{tag}.pth"
    torch.save(payload, pth_path)

    # JSON card without tensors
    card = {k: v for k, v in payload.items() if k != "state_dict"}
    card["artifact"] = {
        "pth": str(pth_path.relative_to(ROOT)).replace("\\", "/"),
        "pth_file_mb": round(pth_path.stat().st_size / (1024**2), 3),
    }
    card_path = out_dir / f"{tag}_model_card.json"
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)

    # Human README
    md_path = out_dir / f"{tag}_MODEL_CARD.md"
    md_path.write_text(_render_md(card), encoding="utf-8")

    # Sidecar config for loaders
    with open(out_dir / f"{tag}_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    print(f"Wrote {pth_path}")
    print(f"Wrote {card_path}")
    print(f"Wrote {md_path}")
    return pth_path


def _render_md(card: dict) -> str:
    ms = card["model_stats"]
    mp = card["metrics_primary"]
    lines = [
        f"# GLAUCO edge model card — `{card['run_name']}`",
        "",
        "**Product:** glaucoma only (binary screening). Deployment is out of scope; this pack is the peak artifact.",
        "",
        "## Architecture & footprint",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Architecture | `{card['architecture']}` |",
        f"| Modality | {card['modality']} |",
        f"| Classes | {card['num_classes']} — {card['class_names']} |",
        f"| Input | {card['input']['channels']}×{card['input']['height']}×{card['input']['width']} NCHW |",
        f"| Parameters | **{ms['parameters_millions']} M** ({ms['parameters_total']:,}) |",
        f"| Weights size | **{ms['weights_mb']} MB** |",
        f"| Pack `.pth` file | {card['artifact']['pth_file_mb']} MB |",
        (
            f"| Peak CUDA (1×infer) | {ms['inference_peak_cuda_mb']} MB |"
            if ms.get("inference_peak_cuda_mb") is not None
            else "| Peak CUDA (1×infer) | n/a (CPU export) |"
        ),
        f"| Decision threshold | {card['threshold']} |",
        "",
        "## Primary metrics (deploy checklist)",
        "",
        f"| Metric | Value |",
        f"|--------|------:|",
        f"| Accuracy | {_fmt(mp.get('accuracy'))} |",
        f"| Precision | {_fmt(mp.get('precision'))} |",
        f"| Recall (=Sensitivity) | {_fmt(mp.get('recall'))} |",
        f"| Sensitivity | {_fmt(mp.get('sensitivity'))} |",
        f"| Specificity | {_fmt(mp.get('specificity'))} |",
        f"| F1 | {_fmt(mp.get('f1'))} |",
        f"| AUC | {_fmt(mp.get('auc'))} |",
        f"| Sens @ 95% Spec | {_fmt(mp.get('sensitivity_at_95_spec'))} |",
        "",
        "## Splits evaluated",
        "",
    ]
    for split, pack in (card.get("metrics_by_split") or {}).items():
        m = pack.get("all", pack)
        lines.append(
            f"- **{split}**: Acc {_fmt(m.get('accuracy'))} · F1 {_fmt(m.get('f1'))} · "
            f"Sens {_fmt(m.get('sensitivity'))} · Spec {_fmt(m.get('specificity'))} · "
            f"AUC {_fmt(m.get('auc'))}"
        )
    lines += [
        "",
        "## Load (deployer)",
        "",
        "```python",
        "import torch",
        f"pack = torch.load('{card['artifact']['pth']}', map_location='cpu', weights_only=False)",
        "state = pack['state_dict']",
        "classes = pack['class_names']  # ['normal', 'glaucoma']",
        "thr = pack['threshold']",
        "stats = pack['model_stats']",
        "```",
        "",
        f"Source train checkpoint: `{card['source_checkpoint']}`",
        f"Exported: {card['exported_at']}",
        "",
    ]
    return "\n".join(lines)


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def _inject_cached_metrics(
    metrics_by_split: dict[str, Any], cached: Path, name: str
) -> None:
    if not cached.exists():
        return
    with open(cached, encoding="utf-8") as f:
        blob = json.load(f)
    # holdout report nests under stem key
    if name in blob and isinstance(blob[name], dict):
        raw = blob[name]
    elif "accuracy" in blob:
        raw = blob
    else:
        # first nested metrics dict
        raw = next((v for v in blob.values() if isinstance(v, dict) and "accuracy" in v), None)
        if raw is None:
            return
    metrics_by_split[name] = enrich_metrics(raw)
    print(f"injected cached metrics → {name}")


def main() -> int:
    p = argparse.ArgumentParser(description="Export GLAUCO edge .pth + metrics card")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints/glauco_retfound_cfp_airogs_light/best.pt",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "exports" / "glauco_edge",
    )
    p.add_argument("--tag", default="glauco_fundus_peak")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--test-csv",
        action="append",
        default=None,
        help="Repeatable. Defaults to AIROGS holdout + key externals.",
    )
    p.add_argument(
        "--cached-holdout",
        type=Path,
        default=ROOT / "checkpoints/glauco_retfound_cfp_airogs_light/eval_holdout_test.json",
        help="If AIROGS images missing, inject this holdout report as primary metrics.",
    )
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    defaults = [
        ROOT / "data/processed/glaucoma/catalogs/airogs_light_holdout_test.csv",
        ROOT / "data/processed/glaucoma/catalogs/airogs_light_test.csv",
        ROOT / "data/processed/glaucoma/catalogs/drishti_full.csv",
        ROOT / "data/processed/glaucoma/catalogs/beh_external.csv",
        ROOT / "data/processed/glaucoma/catalogs/acrima_external.csv",
        ROOT / "data/processed/glaucoma/catalogs/rim_one_hospital_test.csv",
    ]
    csvs = [Path(c) for c in args.test_csv] if args.test_csv else defaults

    # Inline hook: after export_pack builds metrics, inject cache if holdout missing.
    # Simpler: monkey-patch via wrapper
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    ckpt_path = ckpt_path.resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    tag = args.tag

    # Duplicate minimal call path with post-inject
    ckpt, cfg = load_checkpoint(ckpt_path, device)
    num_classes = 2 if bool(cfg.get("binary", True)) else int(cfg.get("num_classes", 2))
    model = build_from_config(cfg, num_classes=num_classes, load_weights=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    stats = build_model_stats(model, cfg, device)

    class Args:
        pass

    a = Args()
    a.batch_size = args.batch_size
    a.image_size = int((cfg.get("data") or {}).get("image_size") or 224)
    a.num_workers = 2
    a.modality = cfg.get("modality", "fundus")
    a.threshold = args.threshold

    metrics_by_split: dict[str, Any] = {}
    for csv in csvs:
        csv = Path(csv)
        if not csv.is_absolute():
            csv = ROOT / csv
        if not csv.exists():
            print(f"skip missing catalog {csv}")
            continue
        import pandas as pd

        sample = pd.read_csv(csv).head(8)
        if "path" in sample.columns and sum(Path(str(p)).exists() for p in sample["path"]) == 0:
            print(f"skip unavailable images for {csv.name}")
            continue
        print(f"eval {csv.name} ...")
        metrics_by_split[csv.stem] = eval_csv(model, cfg, csv, a, device)

    cached = Path(args.cached_holdout) if args.cached_holdout else None
    if cached and cached.exists() and cached.resolve().parent == ckpt_path.parent:
        # Only inject cache from the same run directory as the checkpoint
        stem = cached.stem
        key = "airogs_light_holdout_test" if "holdout" in stem else stem
        if key not in metrics_by_split and "best_metrics" in stem:
            key = "train_best_or_val"
        if all(
            k not in metrics_by_split
            for k in ("airogs_light_holdout_test", "train_best_or_val", cached.stem)
        ):
            _inject_cached_metrics(metrics_by_split, cached, key)

    # Prefer official holdout / first live eval as primary
    ordered: dict[str, Any] = {}
    for pref in (
        "airogs_light_holdout_test",
        "harvard_gd_test",
        "harvard_gf_test",
        "acrima_test",
    ):
        if pref in metrics_by_split:
            ordered[pref] = metrics_by_split.pop(pref)
            break
    ordered.update(metrics_by_split)
    metrics_by_split = ordered

    primary = next(iter(metrics_by_split.values()), {})
    primary_flat = primary.get("all", primary) if isinstance(primary, dict) else {}
    # Fill sens/spec from confusion if missing (older train metrics)
    if primary_flat.get("sensitivity") is None and primary_flat.get("confusion_matrix"):
        cm = primary_flat["confusion_matrix"]
        try:
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            primary_flat["sensitivity"] = tp / max(tp + fn, 1)
            primary_flat["specificity"] = tn / max(tn + fp, 1)
            primary_flat["recall"] = primary_flat["sensitivity"]
            primary_flat["precision"] = tp / max(tp + fp, 1)
        except Exception:
            pass

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": "glauco_edge_v1",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "task": "glaucoma_binary_screening",
        "product": "GLAUCO",
        "source_checkpoint": str(ckpt_path.relative_to(ROOT)).replace("\\", "/"),
        "run_name": cfg.get("run_name") or ckpt_path.parent.name,
        "architecture": stats["architecture"],
        "modality": stats["modality"],
        "num_classes": stats["num_classes"],
        "class_names": stats["class_names"] or CLASS_NAMES_BINARY,
        "class_index": {0: "normal", 1: "glaucoma"},
        "input": stats["input"],
        "threshold": float(args.threshold),
        "model_stats": {
            "parameters_total": stats["parameters"]["total"],
            "parameters_trainable": stats["parameters"]["trainable"],
            "parameters_millions": stats["parameters_millions"],
            "weights_mb": stats["weights_mb"],
            "weights_bytes": stats["weights_bytes"],
            "inference_peak_cuda_mb": stats["inference_peak_cuda_mb"],
        },
        "metrics_primary": {
            "accuracy": primary_flat.get("accuracy"),
            "precision": primary_flat.get("precision") or primary_flat.get("ppv"),
            "recall": primary_flat.get("recall") or primary_flat.get("sensitivity"),
            "sensitivity": primary_flat.get("sensitivity"),
            "specificity": primary_flat.get("specificity"),
            "f1": primary_flat.get("f1"),
            "auc": primary_flat.get("auc"),
            "sensitivity_at_95_spec": primary_flat.get("sensitivity_at_95_spec"),
        },
        "metrics_by_split": metrics_by_split,
        "train_metrics_at_best": ckpt.get("metrics"),
        "config": cfg,
        "state_dict": model.state_dict(),
    }
    pth_path = out_dir / f"{tag}.pth"
    torch.save(payload, pth_path)
    card = {k: v for k, v in payload.items() if k != "state_dict"}
    card["artifact"] = {
        "pth": str(pth_path.relative_to(ROOT)).replace("\\", "/"),
        "pth_file_mb": round(pth_path.stat().st_size / (1024**2), 3),
    }
    card_path = out_dir / f"{tag}_model_card.json"
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    md_path = out_dir / f"{tag}_MODEL_CARD.md"
    md_path.write_text(_render_md(card), encoding="utf-8")
    with open(out_dir / f"{tag}_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    print(f"Wrote {pth_path}")
    print(f"Wrote {card_path}")
    print(f"Wrote {md_path}")
    print(json.dumps({"model_stats": payload["model_stats"], "metrics_primary": payload["metrics_primary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
