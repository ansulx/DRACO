#!/usr/bin/env python3
"""Render docs/demo_oct_20s.html to an MP4 (headless Chromium + imageio)."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "docs" / "demo_oct_20s.html"
OUT_DEFAULT = ROOT / "docs" / "DRACO_OCT_demo_20s.mp4"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render DRACO OCT demo video")
    parser.add_argument("--html", type=Path, default=HTML)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    if not args.html.exists():
        print(f"Missing: {args.html}", file=sys.stderr)
        return 1

    try:
        from playwright.sync_api import sync_playwright
        import imageio.v2 as imageio
        import numpy as np
    except ImportError as e:
        print("Install: pip install playwright pillow imageio imageio-ffmpeg", file=sys.stderr)
        print(e, file=sys.stderr)
        return 1

    n_frames = int(args.duration * args.fps)
    url = args.html.resolve().as_uri()

    frames_dir = Path(tempfile.mkdtemp(prefix="draco_demo_"))
    print(f"Capturing {n_frames} frames from {url}")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": args.width, "height": args.height})
            page.goto(url, wait_until="networkidle")

            # Drive animation from t=0 (reload ensures clean state)
            page.evaluate("() => { window.__demoT0 = performance.now(); }")

            for i in range(n_frames):
                ms = (i / args.fps) * 1000.0
                page.evaluate(
                    """(ms) => {
                      const els = document.querySelectorAll('.reveal');
                      const END = 20000;
                      const bar = document.getElementById('bar');
                      const p = Math.max(0, Math.min(1, ms / END));
                      if (bar) bar.style.width = (p * 100).toFixed(2) + '%';
                      for (const el of els) {
                        const show = Number(el.getAttribute('data-show') || 0);
                        const hide = Number(el.getAttribute('data-hide') || END);
                        el.classList.toggle('show', ms >= show && ms < hide);
                      }
                    }""",
                    ms,
                )
                path = frames_dir / f"frame_{i:04d}.png"
                page.locator("#frame").screenshot(path=str(path))

            browser.close()

        print(f"Encoding {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            str(args.out),
            fps=args.fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
        )
        for i in range(n_frames):
            path = frames_dir / f"frame_{i:04d}.png"
            img = imageio.imread(path)
            writer.append_data(np.asarray(img))
        writer.close()

        size_mb = args.out.stat().st_size / 1e6
        print(f"Done: {args.out} ({size_mb:.1f} MB)")
        return 0
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
