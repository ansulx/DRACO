#!/usr/bin/env python3
"""Build GLAUCO Lucida-style AR demo: 3D OCT showcase + USDZ for iPad Quick Look.

Outputs:
  docs/glauco_ar_demo/
    index.html
    README_IPAD.md
    cases/<id>/...
    usdz/*.usdz
"""

from __future__ import annotations

import json
import math
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "docs" / "glauco_ar_demo"
CASES = OUT / "cases"
USDZ_DIR = OUT / "usdz"

CKPT = ROOT / "checkpoints" / "glauco_efficientnet_harvard_gf_full" / "best.pt"
GF_TEST = ROOT / "data" / "processed" / "glaucoma" / "catalogs" / "harvard_gf_test.csv"
GF_TRAIN = ROOT / "data" / "processed" / "glaucoma" / "catalogs" / "harvard_gf_train.csv"


def _rnflt_to_rgb(rnflt: np.ndarray) -> np.ndarray:
    arr = np.asarray(rnflt, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0)
    lo, hi = 0.0, 350.0
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    try:
        import matplotlib

        rgb = (matplotlib.colormaps["viridis"](norm)[..., :3] * 255).astype(np.uint8)
    except Exception:
        g = (norm * 255).astype(np.uint8)
        rgb = np.stack([g, g, g], axis=-1)
    return rgb


def _save_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _heightfield_mesh(rnflt: np.ndarray, scale_xy: float = 1.0, scale_z: float = 0.35):
    """Build a terrain mesh from RNFLT map for AR (fast vectorized extrude)."""
    from skimage import measure

    z = np.nan_to_num(rnflt.astype(np.float32), nan=0.0)
    z = np.clip(z, 0, 350) / 350.0
    z = z[::2, ::2]
    h, w = z.shape
    depth = 32
    thickness = np.clip(np.round(z * (depth - 1)).astype(np.int32), 1, depth - 1)
    vol = np.zeros((h, w, depth), dtype=np.float32)
    zz = np.arange(depth, dtype=np.int32)[None, None, :]
    vol[zz <= thickness[:, :, None]] = 1.0
    verts, faces, normals, _ = measure.marching_cubes(vol, level=0.5, step_size=1)
    verts = verts.astype(np.float32)
    verts -= verts.mean(axis=0)
    extent = np.abs(verts).max() + 1e-6
    verts *= 0.9 / extent
    colors = np.zeros((len(verts), 3), dtype=np.float32)
    t = (verts[:, 2] - verts[:, 2].min()) / max(float(np.ptp(verts[:, 2])), 1e-6)
    colors[:, 0] = 0.05 + 0.25 * t
    colors[:, 1] = 0.50 + 0.40 * t
    colors[:, 2] = 0.70 + 0.20 * (1 - t)
    return verts, faces.astype(np.int32), colors


def _oct_shell_mesh(oct: np.ndarray):
    """Isosurface of OCT volume (downsampled)."""
    from skimage import measure
    from scipy.ndimage import gaussian_filter

    vol = oct.astype(np.float32)
    # take central crop / downsample
    vol = vol[::2, ::2, ::2]
    vol = gaussian_filter(vol, sigma=1.0)
    thr = float(np.percentile(vol, 70))
    try:
        verts, faces, normals, _ = measure.marching_cubes(vol, level=thr, step_size=2)
    except Exception:
        thr = float(vol.mean() + 0.2 * vol.std())
        verts, faces, normals, _ = measure.marching_cubes(vol, level=thr, step_size=2)
    verts = verts.astype(np.float32)
    verts -= verts.mean(axis=0)
    extent = np.abs(verts).max() + 1e-6
    verts *= 0.9 / extent
    colors = np.tile(np.array([[0.75, 0.82, 0.88]], dtype=np.float32), (len(verts), 1))
    return verts, faces.astype(np.int32), colors


def _write_usdz(path: Path, verts: np.ndarray, faces: np.ndarray, colors: np.ndarray, name: str) -> None:
    """Write a minimal Apple-compatible USDZ (zipped USDA + no textures)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    usda_name = f"{name}.usda"
    # USDA mesh
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "GlaucomaDemo"',
        '    metersPerUnit = 1',
        '    upAxis = "Y"',
        ")",
        "",
        'def Xform "GlaucomaDemo" (',
        '    kind = "component"',
        ")",
        "{",
        '    def Mesh "Surface"',
        "    {",
        "        float3[] extent = [(-1,-1,-1), (1,1,1)]",
        f"        int[] faceVertexCounts = [{', '.join(['3'] * len(faces))}]",
        "        int[] faceVertexIndices = [" + ", ".join(str(int(i)) for i in faces.reshape(-1)) + "]",
        "        point3f[] points = ["
        + ", ".join(f"({v[0]:.6f}, {v[2]:.6f}, {v[1]:.6f})" for v in verts)  # Y-up
        + "]",
        "        color3f[] primvars:displayColor = ["
        + ", ".join(f"({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})" for c in colors)
        + "] (",
        '            interpolation = "vertex"',
        "        )",
        "        uniform token subdivisionScheme = \"none\"",
        "    }",
        "}",
        "",
    ]
    usda_bytes = "\n".join(lines).encode("utf-8")
    # USDZ = zip store (no compression preferred by Apple for some assets; deflated usually OK)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(usda_name, usda_bytes)


def _export_slices(oct: np.ndarray, out_dir: Path) -> dict:
    """Save axial/sagittal/coronal mid slices + a few neighbors for scroll."""
    out_dir.mkdir(parents=True, exist_ok=True)
    d, h, w = oct.shape
    ax_dir = out_dir / "axial"
    sg_dir = out_dir / "sagittal"
    cr_dir = out_dir / "coronal"
    for p in (ax_dir, sg_dir, cr_dir):
        p.mkdir(exist_ok=True)

    def norm_u8(sl):
        sl = sl.astype(np.float32)
        lo, hi = np.percentile(sl, (1, 99))
        sl = np.clip((sl - lo) / max(hi - lo, 1e-6), 0, 1)
        return (sl * 255).astype(np.uint8)

    # save every 4th slice for lighter demo
    ax_idx = list(range(0, d, 4))
    for i in ax_idx:
        _save_png(norm_u8(oct[i]), ax_dir / f"{i:03d}.png")
    sg_idx = list(range(0, w, 4))
    for i in sg_idx:
        _save_png(norm_u8(oct[:, :, i]), sg_dir / f"{i:03d}.png")
    cr_idx = list(range(0, h, 4))
    for i in cr_idx:
        _save_png(norm_u8(oct[:, i, :]), cr_dir / f"{i:03d}.png")

    return {
        "axial": [f"axial/{i:03d}.png" for i in ax_idx],
        "sagittal": [f"sagittal/{i:03d}.png" for i in sg_idx],
        "coronal": [f"coronal/{i:03d}.png" for i in cr_idx],
        "shape": [int(d), int(h), int(w)],
    }


def _pick_cases(n_each: int = 2) -> list[dict]:
    import pandas as pd

    frames = []
    for p in (GF_TEST, GF_TRAIN):
        if p.exists():
            frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    cases = []
    for label, tag in [(1, "glaucoma"), (0, "normal")]:
        sub = df[df["label"].astype(int) == label].sample(
            n=min(n_each, (df["label"].astype(int) == label).sum()),
            random_state=7,
        )
        for _, row in sub.iterrows():
            cases.append(
                {
                    "path": Path(row["path"]),
                    "label": int(row["label"]),
                    "tag": tag,
                    "race": str(row.get("race", "")),
                    "gender": str(row.get("gender", "")),
                }
            )
    return cases


def _predict_rnflt(rnflt: np.ndarray, model, device) -> dict:
    from glauco.data.datasets import _array_to_rgb
    from glauco.data.transforms import build_transforms

    img = _array_to_rgb(rnflt)
    tfm = build_transforms("eval", 224, "rnflt")
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
    return {
        "prob_normal": float(prob[0]),
        "prob_glaucoma": float(prob[1]),
        "pred": int(prob.argmax()),
        "risk_score_1_5": round(1.0 + 4.0 * float(prob[1]), 2),
    }


def _load_model(device):
    from glauco.models import build_from_config
    from glauco.evaluate import load_checkpoint

    ckpt, cfg = load_checkpoint(CKPT, device)
    model = build_from_config(cfg, num_classes=2, load_weights=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg


def build_case(meta: dict, model, device, idx: int) -> dict:
    path = meta["path"]
    with np.load(path, allow_pickle=False) as data:
        oct = data["oct_bscans"]
        rnflt = data["rnflt"]
        age = float(data["age"]) if "age" in data else None
        glaucoma = int(data["glaucoma"])

    case_id = f"case_{idx:02d}_{meta['tag']}"
    cdir = CASES / case_id
    cdir.mkdir(parents=True, exist_ok=True)

    # RNFLT image
    rgb = _rnflt_to_rgb(rnflt)
    _save_png(rgb, cdir / "rnflt.png")

    # slices
    slice_info = _export_slices(oct, cdir / "slices")

    # AI
    pred = _predict_rnflt(rnflt, model, device)

    # meshes → USDZ
    try:
        v, f, col = _heightfield_mesh(rnflt)
        usdz_rnflt = USDZ_DIR / f"{case_id}_rnflt_surface.usdz"
        _write_usdz(usdz_rnflt, v, f, col, f"{case_id}_rnflt")
    except Exception as e:
        print(f"RNFLT mesh failed {case_id}: {e}")
        usdz_rnflt = None

    try:
        v2, f2, c2 = _oct_shell_mesh(oct)
        usdz_oct = USDZ_DIR / f"{case_id}_oct_shell.usdz"
        _write_usdz(usdz_oct, v2, f2, c2, f"{case_id}_oct")
    except Exception as e:
        print(f"OCT mesh failed {case_id}: {e}")
        usdz_oct = None

    # also save GLB via trimesh for web 3D fallback
    try:
        import trimesh

        v, f, col = _heightfield_mesh(rnflt)
        mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
        mesh.visual.vertex_colors = (col * 255).astype(np.uint8)
        mesh.export(cdir / "rnflt_surface.glb")
    except Exception as e:
        print(f"GLB export failed: {e}")

    info = {
        "id": case_id,
        "source_file": path.name,
        "truth_label": glaucoma,
        "truth_name": "glaucoma" if glaucoma else "normal",
        "age": age,
        "race": meta.get("race"),
        "gender": meta.get("gender"),
        "ai": pred,
        "rnflt_png": "rnflt.png",
        "slices": slice_info,
        "usdz_rnflt": f"../usdz/{usdz_rnflt.name}" if usdz_rnflt else None,
        "usdz_oct": f"../usdz/{usdz_oct.name}" if usdz_oct else None,
        "glb": "rnflt_surface.glb",
        "volume_ml_proxy": round(float(np.nanmean(np.clip(rnflt, 0, 350))) * 0.02, 2),
    }
    (cdir / "meta.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Built {case_id}: truth={info['truth_name']} AI risk={pred['risk_score_1_5']}")
    return info


def write_html(manifest: list[dict]) -> None:
    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>GLAUCO · Clinical 3D Demo</title>
<script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
<style>
  :root {
    --bg:#0b1220; --panel:#121a2b; --ink:#e8eef7; --muted:#8aa0b8;
    --teal:#2ec4b6; --line:#243047; --good:#3ddc97; --warn:#ffb703; --bad:#ef476f;
  }
  * { box-sizing:border-box; }
  body {
    margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    background:radial-gradient(1200px 600px at 10% -10%, #16324f 0%, var(--bg) 55%);
    color:var(--ink);
  }
  header {
    display:flex; justify-content:space-between; align-items:center; gap:12px;
    padding:14px 18px; border-bottom:1px solid var(--line); background:rgba(10,16,28,.85);
    backdrop-filter: blur(10px); position:sticky; top:0; z-index:10;
  }
  header h1 { margin:0; font-size:18px; letter-spacing:.02em; }
  header .sub { color:var(--muted); font-size:12px; }
  .wrap { display:grid; grid-template-columns: 1fr 320px; gap:12px; padding:12px; min-height:calc(100vh - 58px); }
  @media (max-width: 980px) { .wrap { grid-template-columns:1fr; } }
  .grid {
    display:grid; grid-template-columns: repeat(3, 1fr); gap:10px;
  }
  @media (max-width: 980px) { .grid { grid-template-columns:1fr 1fr; } }
  .tile {
    background:var(--panel); border:1px solid var(--line); border-radius:14px; overflow:hidden;
    min-height:220px; display:flex; flex-direction:column;
  }
  .tile .lab {
    display:flex; justify-content:space-between; align-items:center;
    padding:8px 10px; font-size:11px; color:var(--muted); border-bottom:1px solid var(--line);
    text-transform:uppercase; letter-spacing:.06em;
  }
  .tile .lab strong { color:var(--ink); font-weight:600; text-transform:none; letter-spacing:0; font-size:12px; }
  .stage { flex:1; display:flex; align-items:center; justify-content:center; background:#070b14; position:relative; }
  .stage img { max-width:100%; max-height:260px; object-fit:contain; }
  input[type=range] { width:100%; }
  .side {
    background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:14px;
    height:fit-content; position:sticky; top:70px;
  }
  .side h2 { margin:0 0 8px; font-size:15px; }
  .kpi { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin:10px 0 14px; }
  .kpi div { background:#0d1524; border:1px solid var(--line); border-radius:10px; padding:10px; }
  .kpi .n { font-size:22px; font-weight:700; color:var(--teal); }
  .kpi .l { font-size:11px; color:var(--muted); margin-top:2px; }
  .bar { height:10px; background:#1a2438; border-radius:99px; overflow:hidden; margin:6px 0 12px; }
  .bar > span { display:block; height:100%; background:linear-gradient(90deg,#2ec4b6,#ffb703,#ef476f); }
  select, button, a.btn {
    width:100%; appearance:none; border:1px solid var(--line); background:#0d1524; color:var(--ink);
    border-radius:10px; padding:10px 12px; font-size:13px; margin:4px 0 8px;
  }
  a.btn { display:block; text-align:center; text-decoration:none; background:linear-gradient(135deg,#1a7a8c,#2ec4b6); border:none; font-weight:600; }
  a.btn.secondary { background:#0d1524; border:1px solid var(--line); font-weight:500; }
  .pill { display:inline-block; padding:3px 8px; border-radius:999px; font-size:11px; font-weight:600; }
  .pill.ok { background:rgba(61,220,151,.15); color:var(--good); }
  .pill.bad { background:rgba(239,71,111,.15); color:var(--bad); }
  .note { font-size:11px; color:var(--muted); line-height:1.45; margin-top:10px; }
  model-viewer { width:100%; height:260px; background:#070b14; }
</style>
</head>
<body>
<header>
  <div>
    <h1>GLAUCO Clinical 3D Demo</h1>
    <div class="sub">Lucida-style workstation · Harvard-GF 200³ OCT · on-device AR via USDZ</div>
  </div>
  <div class="sub">Research demo · not a medical device</div>
</header>
<div class="wrap">
  <div>
    <div class="grid">
      <div class="tile">
        <div class="lab"><strong>OCT Axial</strong><span id="axLab">—</span></div>
        <div class="stage"><img id="axImg" alt="axial"/></div>
        <div style="padding:8px 10px"><input id="axRange" type="range" min="0" max="0" value="0"/></div>
      </div>
      <div class="tile">
        <div class="lab"><strong>OCT Sagittal</strong><span id="sgLab">—</span></div>
        <div class="stage"><img id="sgImg" alt="sagittal"/></div>
        <div style="padding:8px 10px"><input id="sgRange" type="range" min="0" max="0" value="0"/></div>
      </div>
      <div class="tile">
        <div class="lab"><strong>OCT Coronal</strong><span id="crLab">—</span></div>
        <div class="stage"><img id="crImg" alt="coronal"/></div>
        <div style="padding:8px 10px"><input id="crRange" type="range" min="0" max="0" value="0"/></div>
      </div>
      <div class="tile">
        <div class="lab"><strong>RNFLT map</strong><span>viridis µm</span></div>
        <div class="stage"><img id="rnfltImg" alt="rnflt"/></div>
      </div>
      <div class="tile" style="grid-column: span 2;">
        <div class="lab"><strong>3D RNFLT surface</strong><span>drag · pinch on iPad</span></div>
        <div class="stage" style="padding:0">
          <model-viewer id="mv" camera-controls touch-action="pan-y" exposure="1.0" shadow-intensity="0.4"></model-viewer>
        </div>
      </div>
    </div>
  </div>
  <aside class="side">
    <h2>GLAUCO Results</h2>
    <label class="sub">Showcase case</label>
    <select id="caseSel"></select>
    <div id="truthPill"></div>
    <div class="kpi">
      <div><div class="n" id="riskN">—</div><div class="l">AI risk (1–5)</div></div>
      <div><div class="n" id="probN">—</div><div class="l">P(glaucoma)</div></div>
      <div><div class="n" id="volN">—</div><div class="l">RNFL proxy index</div></div>
      <div><div class="n" id="ageN">—</div><div class="l">Age (yrs)</div></div>
    </div>
    <div class="sub">Glaucoma probability</div>
    <div class="bar"><span id="probBar" style="width:0%"></span></div>
    <div class="sub" id="metaLine">—</div>
    <a class="btn" id="usdzRnflt" href="#" rel="ar">Open RNFLT in AR (USDZ)</a>
    <a class="btn secondary" id="usdzOct" href="#" rel="ar">Open OCT shell in AR</a>
    <p class="note">
      On <b>iPhone/iPad</b>: tap USDZ → AR Quick Look.<br/>
      On <b>Mac</b>: open USDZ in Preview / Quick Look.<br/>
      Model: EfficientNet RNFLT (Harvard-GF checkpoint). Multi-planar OCT from real <code>oct_bscans (200³)</code>.
    </p>
  </aside>
</div>
<script>
const MANIFEST = __MANIFEST__;
const $ = (id) => document.getElementById(id);
let cur = null;

function fillSelect() {
  const sel = $('caseSel');
  MANIFEST.forEach((c, i) => {
    const o = document.createElement('option');
    o.value = i;
    o.textContent = `${c.id} · truth=${c.truth_name} · AI=${c.ai.risk_score_1_5}`;
    sel.appendChild(o);
  });
  sel.onchange = () => loadCase(MANIFEST[+sel.value]);
}

function bindStack(rangeId, imgId, labId, files, base) {
  const r = $(rangeId), img = $(imgId), lab = $(labId);
  r.max = Math.max(0, files.length - 1);
  r.value = Math.floor(files.length / 2);
  const sync = () => {
    const i = +r.value;
    img.src = base + '/' + files[i];
    lab.textContent = `${i+1}/${files.length}`;
  };
  r.oninput = sync; sync();
}

function loadCase(c) {
  cur = c;
  const base = 'cases/' + c.id;
  $('rnfltImg').src = base + '/' + c.rnflt_png;
  bindStack('axRange','axImg','axLab', c.slices.axial, base + '/slices');
  bindStack('sgRange','sgImg','sgLab', c.slices.sagittal, base + '/slices');
  bindStack('crRange','crImg','crLab', c.slices.coronal, base + '/slices');
  $('mv').src = base + '/' + c.glb;
  $('riskN').textContent = c.ai.risk_score_1_5.toFixed(1);
  $('probN').textContent = (c.ai.prob_glaucoma*100).toFixed(1) + '%';
  $('probBar').style.width = (c.ai.prob_glaucoma*100).toFixed(1) + '%';
  $('volN').textContent = c.volume_ml_proxy;
  $('ageN').textContent = c.age ? c.age.toFixed(1) : '—';
  const ok = c.ai.pred === c.truth_label;
  $('truthPill').innerHTML = `<span class="pill ${c.truth_label? 'bad':'ok'}">Truth: ${c.truth_name}</span>
    <span class="pill ${ok?'ok':'bad'}" style="margin-left:6px">AI: ${c.ai.pred? 'glaucoma':'normal'} ${ok?'✓':'✗'}</span>`;
  $('metaLine').textContent = `${c.race || ''} · ${c.gender || ''} · ${c.source_file}`;
  const a1 = $('usdzRnflt'), a2 = $('usdzOct');
  if (c.usdz_rnflt) { a1.href = c.usdz_rnflt.replace('../',''); a1.style.display='block'; } else a1.style.display='none';
  if (c.usdz_oct) { a2.href = c.usdz_oct.replace('../',''); a2.style.display='block'; } else a2.style.display='none';
}

fillSelect();
loadCase(MANIFEST[0]);
</script>
</body>
</html>
"""
    html = html.replace("__MANIFEST__", json.dumps(manifest))
    (OUT / "index.html").write_text(html, encoding="utf-8")


def write_readme() -> None:
    text = """# GLAUCO AR / 3D Demo (iPad · MacBook)

Lucida-style multi-planar OCT viewer + AI risk panel + **USDZ** for Apple AR Quick Look.

## On iPad / iPhone
1. Copy the whole `glauco_ar_demo` folder to the iPad (Files app / AirDrop / iCloud).
2. Open `index.html` in **Safari** (local file or via a tiny static server).
3. Tap **Open RNFLT in AR (USDZ)** → Apple **AR Quick Look** places the 3D surface in your room.
4. Pinch / drag the in-page 3D model for non-AR viewing.

If Safari blocks local modules, from Mac terminal in this folder:
```bash
python3 -m http.server 8000
```
Then on iPad open `http://<mac-lan-ip>:8000`.

## On MacBook
- Double-click any `.usdz` in `usdz/` → Quick Look / Preview.
- Or open `index.html` in Safari / Chrome.

## What’s real
- OCT volumes: Harvard-GF `oct_bscans` **200×200×200**
- RNFLT maps from same `.npz`
- AI scores: EfficientNet checkpoint `glauco_efficientnet_harvard_gf_full`
- USDZ: RNFLT thickness surface (+ OCT isosurface shell)

## Disclaimer
Research / presentation demo only — **not** a medical device, not CE/FDA cleared.
"""
    (OUT / "README_IPAD.md").write_text(text, encoding="utf-8")


def main() -> int:
    if not CKPT.exists():
        raise SystemExit(f"Missing checkpoint {CKPT}")
    OUT.mkdir(parents=True, exist_ok=True)
    CASES.mkdir(parents=True, exist_ok=True)
    USDZ_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, _ = _load_model(device)
    cases = _pick_cases(2)
    manifest = []
    for i, meta in enumerate(cases, 1):
        manifest.append(build_case(meta, model, device, i))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_html(manifest)
    write_readme()
    print(f"\nDemo ready: {OUT}")
    print(f"USDZ files: {list(USDZ_DIR.glob('*.usdz'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
