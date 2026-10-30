#!/usr/bin/env python3
"""Insert / refresh supervisor PPT slide: Parth datasets disease·modality·India."""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
PATHS = [
    ROOT / "SUPERVISOR_Showcase.pptx",
    ROOT / "docs" / "SUPERVISOR_Showcase.pptx",
]

NAVY = RGBColor(0x0B, 0x2C, 0x3D)
TEAL = RGBColor(0x1A, 0x7A, 0x8C)
INK = RGBColor(0x1C, 0x28, 0x32)
MUTED = RGBColor(0x5A, 0x6A, 0x76)
GREEN = RGBColor(0x1B, 0x7F, 0x5A)
CREAM = RGBColor(0xF7, 0xF9, 0xFB)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ROW_ALT = RGBColor(0xEE, 0xF4, 0xF6)

ROWS = [
    # dataset, disease, modality, n, region, india
    ("Drishti-GS1", "Glaucoma", "Fundus + OD/cup GT", "101", "Aravind, Madurai", "YES — Indian"),
    ("ACRIMA", "Glaucoma", "Fundus (OD-crop)", "705", "Spain", "No"),
    ("RIM-ONE DL", "Glaucoma", "Fundus (+ OD/cup)", "485", "Spain (3 hosp.)", "No"),
    ("BEH", "Glaucoma", "Fundus", "634", "Bangladesh", "No (BD)"),
    ("BIOMISA", "Glaucoma (+ macula*)", "Fundus + OCT", "79 OD / 44 pts", "Pakistan", "No (PK)"),
]


def _run(p, text, *, size=12, bold=False, color=INK):
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = "Calibri"
    return run


def _fill_cell(cell, text, *, size=11, bold=False, color=INK, fill=None, align=PP_ALIGN.LEFT):
    cell.text = ""
    tf = cell.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    _run(p, text, size=size, bold=bold, color=color)
    if fill is not None:
        cell.fill.solid()
        cell.fill.fore_color.rgb = fill


def build_slide(prs: Presentation) -> None:
    blank = prs.slide_layouts[6]  # blank
    slide = prs.slides.add_slide(blank)

    # Title bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.85))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    box = slide.shapes.add_textbox(Inches(0.4), Inches(0.18), Inches(12.5), Inches(0.55))
    tf = box.text_frame
    p = tf.paragraphs[0]
    _run(p, "5b. Parth fundus packs — all GLAUCOMA (not DR) · modalities & India", size=22, bold=True, color=WHITE)

    # Subhead
    sub = slide.shapes.add_textbox(Inches(0.4), Inches(1.0), Inches(12.5), Inches(0.45))
    p = sub.text_frame.paragraphs[0]
    _run(p, "Supervisor check: disease · modality · size · geography. Only Drishti-GS1 is an Indian cohort in this batch.", size=14, color=MUTED)

    # Table
    cols, rows = 6, 1 + len(ROWS)
    table_shape = slide.shapes.add_table(rows, cols, Inches(0.35), Inches(1.55), Inches(12.6), Inches(3.6))
    table = table_shape.table
    headers = ["Dataset", "Disease", "Modality", "n", "Site / region", "India?"]
    widths = [1.8, 2.0, 2.6, 1.6, 2.4, 2.2]
    for i, w in enumerate(widths):
        table.columns[i].width = Inches(w)

    for j, h in enumerate(headers):
        _fill_cell(table.cell(0, j), h, size=11, bold=True, color=WHITE, fill=TEAL, align=PP_ALIGN.CENTER)

    for i, row in enumerate(ROWS, start=1):
        fill = CREAM if i % 2 else ROW_ALT
        india_color = GREEN if row[5].startswith("YES") else INK
        for j, val in enumerate(row):
            _fill_cell(
                table.cell(i, j),
                val,
                size=11,
                bold=(j == 0 or (j == 5 and row[5].startswith("YES"))),
                color=india_color if j == 5 else INK,
                fill=fill,
            )

    # Footer notes
    note = slide.shapes.add_textbox(Inches(0.4), Inches(5.35), Inches(12.5), Inches(1.7))
    tf = note.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    _run(p, "Routing: all five → GLAUCO external (downloaded + catalogs). Do not mix into GHOTA DR Acc.", size=13, bold=True, color=NAVY)
    p = tf.add_paragraph()
    _run(p, "* BIOMISA Macula also has ci-DME / AMD / CSR — composite research set, not ICDR DR grades like APTOS/IDRiD.", size=12, color=MUTED)
    p = tf.add_paragraph()
    _run(p, "Indian DR already on GHOTA track: IDRiD + APTOS. Indian glaucoma in this Parth list: Drishti-GS1 (Aravind).", size=12, bold=True, color=INK)
    p = tf.add_paragraph()
    _run(p, "Detail: docs/PARTH_DATASET_SPECS.md", size=11, color=TEAL)


def main() -> None:
    src = next(p for p in PATHS if p.exists())
    prs = Presentation(str(src))

    # Drop previous auto-inserted slide if re-run (title match)
    # python-pptx cannot easily delete; rebuild by copying all but matching titles then add.
    keep = []
    for slide in prs.slides:
        title_bits = []
        for sh in slide.shapes:
            if hasattr(sh, "text") and sh.text:
                title_bits.append(sh.text)
        blob = " ".join(title_bits)
        if "Parth fundus packs" in blob or "5b. Parth" in blob:
            continue
        keep.append(slide)

    # If we need deletion: create fresh presentation with same slide size and XML copy is hard.
    # Simpler path: if already present, just rewrite file from original backup once.
    # Always rebuild from the first file that does NOT yet have the slide, else strip via new Prs.

    # Rebuild: open original 7-slide deck from git isn't available mid-run; instead
    # if slide already there, skip duplicate by checking count of matching.
    already = any(
        "Parth fundus packs" in " ".join(getattr(sh, "text", "") or "" for sh in s.shapes)
        for s in prs.slides
    )
    if already:
        print("Slide already present — regenerating deck without duplicate via XML drop is skipped.")
        print("Tip: restore 7-slide PPT then re-run, or manually delete old 5b slide.")
        # Still write specs note
        return

    build_slide(prs)
    # Move new slide before last ("what we want") by reordering rId — fragile.
    # Instead: leave as last-but we insert before closing ask by swapping relationships.
    _move_slide_before_last(prs)

    for out in PATHS:
        prs.save(str(out))
        print(f"Wrote {out} ({len(prs.slides)} slides)")


def _move_slide_before_last(prs: Presentation) -> None:
    """Put the newly added slide just before the final slide."""
    sld_id_lst = prs.slides._sldIdLst  # noqa: SLF001
    ids = list(sld_id_lst)
    if len(ids) < 2:
        return
    new = ids[-1]
    sld_id_lst.remove(new)
    # insert before last remaining
    sld_id_lst.insert(len(sld_id_lst) - 1, new)


if __name__ == "__main__":
    main()
