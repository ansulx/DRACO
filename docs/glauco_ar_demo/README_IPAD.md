# GLAUCO AR / 3D Demo (iPad · MacBook)

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
