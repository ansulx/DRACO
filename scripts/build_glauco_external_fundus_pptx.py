#!/usr/bin/env python3
"""Build GLAUCO external fundus dataset review PPT for supervisors."""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt, Emu

ROOT = Path(__file__).resolve().parents[1]
OUTS = [
    ROOT / "docs" / "GLAUCO_External_Fundus_Datasets.pptx",
    ROOT / "GLAUCO_External_Fundus_Datasets.pptx",
]

NAVY = RGBColor(0x0B, 0x2C, 0x3D)
TEAL = RGBColor(0x1A, 0x7A, 0x8C)
TEAL_LT = RGBColor(0x2A, 0x9D, 0xB0)
CREAM = RGBColor(0xF4, 0xF7, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x1C, 0x28, 0x32)
MUTED = RGBColor(0x5A, 0x6A, 0x76)
GREEN = RGBColor(0x1B, 0x7F, 0x5A)
AMBER = RGBColor(0xB0, 0x75, 0x1A)
RED = RGBColor(0xA0, 0x3A, 0x3A)
ROW_A = RGBColor(0xFF, 0xFF, 0xFF)
ROW_B = RGBColor(0xE8, 0xF2, 0xF4)


def _font(run, *, size=14, bold=False, color=INK, name="Calibri"):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = name


def _fill(shape, color: RGBColor):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def _bg(slide):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    _fill(sh, CREAM)
    # subtle left accent
    accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.08), Inches(7.5))
    _fill(accent, TEAL)


def _banner(slide, title: str, subtitle: str):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.05))
    _fill(bar, NAVY)
    # teal underline
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(1.05), Inches(13.333), Inches(0.06))
    _fill(line, TEAL)
    tb = slide.shapes.add_textbox(Inches(0.45), Inches(0.2), Inches(12.4), Inches(0.45))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run()
    r.text = title
    _font(r, size=24, bold=True, color=WHITE)
    sb = slide.shapes.add_textbox(Inches(0.45), Inches(0.62), Inches(12.4), Inches(0.35))
    p = sb.text_frame.paragraphs[0]
    r = p.add_run()
    r.text = subtitle
    _font(r, size=12, color=TEAL_LT)


def _textbox(slide, left, top, width, height, text, *, size=14, bold=False, color=INK, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    _font(r, size=size, bold=bold, color=color)
    return box


def _cell_fill(cell, color: RGBColor):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    for child in list(tcPr):
        if "solidFill" in child.tag:
            tcPr.remove(child)
    solid = tcPr.makeelement(qn("a:solidFill"), {})
    srgb = solid.makeelement(qn("a:srgbClr"), {"val": f"{color[0]:02X}{color[1]:02X}{color[2]:02X}"})
    solid.append(srgb)
    tcPr.append(solid)


def _set_cell(cell, text, *, size=11, bold=False, color=INK, fill=None, align=PP_ALIGN.LEFT):
    cell.text = ""
    tf = cell.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = str(text)
    _font(r, size=size, bold=bold, color=color)
    if fill is not None:
        _cell_fill(cell, fill)
    # vertical center
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE


def _table(slide, left, top, width, height, data, col_widths, *, header=True):
    rows, cols = len(data), len(data[0])
    shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    table = shape.table
    for i, w in enumerate(col_widths):
        table.columns[i].width = w
    for r, row in enumerate(data):
        for c, val in enumerate(row):
            is_h = header and r == 0
            fill = NAVY if is_h else (ROW_B if r % 2 else ROW_A)
            color = WHITE if is_h else INK
            # highlight India / YES cells lightly via bold already
            _set_cell(
                table.cell(r, c),
                val,
                size=10 if not is_h else 11,
                bold=is_h or c == 0,
                color=color,
                fill=fill,
                align=PP_ALIGN.CENTER if c > 0 else PP_ALIGN.LEFT,
            )
    return shape


def _kpi(slide, left, top, value, label, *, accent=TEAL):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, Inches(2.85), Inches(1.15))
    _fill(card, WHITE)
    card.line.color.rgb = accent
    card.line.width = Pt(1.25)
    _textbox(slide, left, top + Inches(0.15), Inches(2.85), Inches(0.5), value, size=22, bold=True, color=accent, align=PP_ALIGN.CENTER)
    _textbox(slide, left, top + Inches(0.65), Inches(2.85), Inches(0.4), label, size=11, color=MUTED, align=PP_ALIGN.CENTER)


def build() -> list[Path]:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # ========== 1 TITLE ==========
    s = prs.slides.add_slide(blank)
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    _fill(bg, NAVY)
    # diagonal teal panel
    panel = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(5.6), Inches(13.333), Inches(1.9))
    _fill(panel, TEAL)
    _textbox(s, Inches(0.7), Inches(1.8), Inches(11.5), Inches(0.5), "GLAUCO  ·  Dataset Review", size=16, bold=True, color=TEAL_LT)
    _textbox(s, Inches(0.7), Inches(2.35), Inches(11.5), Inches(1.0), "External Fundus Datasets — Review & Adoption", size=32, bold=True, color=WHITE)
    _textbox(
        s,
        Inches(0.7),
        Inches(3.5),
        Inches(11.5),
        Inches(0.8),
        "Disease · modality · geography · zero-shot external · ACRIMA adaptation\nAdoption recommendation for GLAUCO (keep GHOTA on DR data only).",
        size=15,
        color=RGBColor(0xC8, 0xD8, 0xDE),
    )
    _textbox(s, Inches(0.7), Inches(6.05), Inches(11.5), Inches(0.9), "OCT / Fundus AI  ·  Supervisor briefing  ·  On disk & experimented", size=14, bold=True, color=WHITE)

    # ========== 2 AGENDA ==========
    s = prs.slides.add_slide(blank)
    _bg(s)
    _banner(s, "Agenda", "What we checked · what we measured · what we recommend")
    items = [
        ("01", "Scope", "Candidate public sets — disease, modality, and product fit"),
        ("02", "Specs & India", "Size, site, and which cohort is Indian"),
        ("03", "Experiments", "AIROGS fundus specialist zero-shot + ACRIMA fine-tune"),
        ("04", "Adoption card", "Clear YES / LATER / NO for GLAUCO vs GHOTA"),
        ("05", "Ask", "Confirm routing and standing external protocol"),
    ]
    for i, (num, title, body) in enumerate(items):
        y = Inches(1.4 + i * 1.05)
        chip = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), y, Inches(1.1), Inches(0.85))
        _fill(chip, NAVY)
        _textbox(s, Inches(0.5), y + Inches(0.22), Inches(1.1), Inches(0.45), num, size=18, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.85), y, Inches(10.8), Inches(0.85))
        _fill(card, WHITE)
        card.line.color.rgb = TEAL_LT
        _textbox(s, Inches(2.1), y + Inches(0.12), Inches(10.2), Inches(0.35), title, size=16, bold=True, color=NAVY)
        _textbox(s, Inches(2.1), y + Inches(0.45), Inches(10.2), Inches(0.35), body, size=13, color=MUTED)

    # ========== 3 SCOPE ==========
    s = prs.slides.add_slide(blank)
    _bg(s)
    _banner(s, "1  ·  Scope — disease & product routing", "All five recent links are glaucoma-side · none replace APTOS/IDRiD for DR Acc")
    left = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.45), Inches(1.4), Inches(6.0), Inches(5.5))
    _fill(left, WHITE)
    left.line.color.rgb = TEAL
    _textbox(s, Inches(0.7), Inches(1.6), Inches(5.5), Inches(0.4), "GLAUCO  (adopt here)", size=18, bold=True, color=TEAL)
    for i, line in enumerate(
        [
            "ACRIMA — fundus, OD-cropped, Spain",
            "BEH — fundus, Bangladesh Eye Hospital",
            "RIM-ONE DL — fundus, 3 Spanish hospitals",
            "Drishti-GS1 — fundus + OD/cup, Aravind India",
            "BIOMISA — fundus + OCT composite (PK)",
        ]
    ):
        _textbox(s, Inches(0.75), Inches(2.2 + i * 0.7), Inches(5.4), Inches(0.55), "▸  " + line, size=14, color=INK)

    right = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.4), Inches(6.0), Inches(5.5))
    _fill(right, WHITE)
    right.line.color.rgb = AMBER
    _textbox(s, Inches(7.05), Inches(1.6), Inches(5.5), Inches(0.4), "GHOTA  (do not mix)", size=18, bold=True, color=AMBER)
    for i, line in enumerate(
        [
            "Keep DR Acc on APTOS + IDRiD (+ OEFI fusion)",
            "BIOMISA ‘ci-DME’ ≠ ICDR 0–4 DR grades",
            "OCTA-500 / ROAD = future OCTA branch",
            "DcardNet = cite only (no public data)",
            "Rule: one disease family per product",
        ]
    ):
        _textbox(s, Inches(7.1), Inches(2.2 + i * 0.7), Inches(5.4), Inches(0.55), "▸  " + line, size=14, color=INK)

    # ========== 4 SPECS TABLE ==========
    s = prs.slides.add_slide(blank)
    _bg(s)
    _banner(s, "2  ·  Dataset specs & geography", "Only Drishti-GS1 is an Indian cohort among these candidates")
    data = [
        ["Dataset", "Disease", "Modality", "n", "Site / region", "India?"],
        ["Drishti-GS1", "Glaucoma", "Fundus + OD/cup GT", "101", "Aravind, Madurai", "YES"],
        ["ACRIMA", "Glaucoma", "Fundus (OD-crop)", "705", "Valencia, Spain", "No"],
        ["RIM-ONE DL", "Glaucoma", "Fundus (+ OD/cup)", "485", "Spain (3 hospitals)", "No"],
        ["BEH", "Glaucoma", "Fundus", "634", "Dhaka, Bangladesh", "No (BD)"],
        ["BIOMISA", "Glaucoma (+ macula)", "Fundus + OCT", "79 OD / 44 pts", "Pakistan (NUST)", "No (PK)"],
    ]
    _table(
        s,
        Inches(0.35),
        Inches(1.4),
        Inches(12.6),
        Inches(3.6),
        data,
        [Inches(1.9), Inches(2.1), Inches(2.5), Inches(1.5), Inches(2.6), Inches(2.0)],
    )
    note = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.35), Inches(5.3), Inches(12.6), Inches(1.7))
    _fill(note, WHITE)
    note.line.color.rgb = GREEN
    _textbox(s, Inches(0.55), Inches(5.45), Inches(12.2), Inches(0.35), "Indian framing for supervisors", size=14, bold=True, color=GREEN)
    _textbox(
        s,
        Inches(0.55),
        Inches(5.9),
        Inches(12.2),
        Inches(0.9),
        "• Indian glaucoma (this batch): Drishti-GS1 only — Aravind Eye Hospital, Madurai.\n"
        "• Indian DR (already on GHOTA): IDRiD + APTOS — separate product track.\n"
        "• South Asia externals: BEH (BD), BIOMISA (PK). EU Acc/adaptation: ACRIMA + RIM-ONE.",
        size=13,
        color=INK,
    )

    # ========== 5 EXPERIMENTS ==========
    s = prs.slides.add_slide(blank)
    _bg(s)
    _banner(s, "3  ·  Experiments — AIROGS fundus specialist", "Model A: zero-shot external  ·  Model B: fine-tune on ACRIMA train")
    _kpi(s, Inches(0.4), Inches(1.35), "0.97", "AIROGS in-domain AUC", accent=TEAL)
    _kpi(s, Inches(3.5), Inches(1.35), "0.95", "Drishti official-test AUC", accent=GREEN)
    _kpi(s, Inches(6.6), Inches(1.35), "0.997", "ACRIMA after FT AUC", accent=TEAL)
    _kpi(s, Inches(9.7), Inches(1.35), "~0.57", "OD-crop zero-shot AUC", accent=RED)

    exp = [
        ["Dataset", "n", "Zero-shot AUC", "After ACRIMA FT", "Interpretation"],
        ["Drishti full (India)", "101", "0.890", "0.886", "Transfers — use as Indian external"],
        ["Drishti official test", "51", "0.951", "0.939", "Strong Acc on official split"],
        ["BEH", "634", "0.782", "0.774", "Usable South-Asia site shift"],
        ["ACRIMA", "705 / 141 test", "0.578", "0.997", "OD-crop gap; use for adaptation"],
        ["RIM-ONE hospital OOD", "174", "0.567", "0.513", "Hard multi-site stress test"],
    ]
    _table(
        s,
        Inches(0.35),
        Inches(2.75),
        Inches(12.6),
        Inches(3.5),
        exp,
        [Inches(2.6), Inches(1.5), Inches(2.0), Inches(2.2), Inches(4.3)],
    )
    _textbox(
        s,
        Inches(0.4),
        Inches(6.5),
        Inches(12.5),
        Inches(0.6),
        "Why ACRIMA / RIM-ONE fail zero-shot: optic-disc crops vs AIROGS full-field fundus — domain gap, not a label bug.",
        size=13,
        bold=True,
        color=MUTED,
    )

    # ========== 6 Adoption card ==========
    s = prs.slides.add_slide(blank)
    _bg(s)
    _banner(s, "4  ·  Adoption decision", "What we recommend adding to the GLAUCO pipeline")
    hire = [
        ["Dataset", "Adopt?", "Role", "Product"],
        ["Drishti-GS1", "YES", "Indian external Acc + ONH/CDR story", "GLAUCO"],
        ["BEH", "YES (light)", "South-Asia site-shift external", "GLAUCO"],
        ["ACRIMA", "YES", "OD-crop adaptation / fine-tune data", "GLAUCO"],
        ["RIM-ONE DL", "YES", "Hard by-hospital domain-shift bench", "GLAUCO"],
        ["BIOMISA", "LATER", "Tiny OD n; optional fundus↔OCT sandbox", "GLAUCO later"],
        ["Any → GHOTA/DR", "NO", "Wrong disease / wrong label scheme", "Keep out"],
    ]
    _table(
        s,
        Inches(0.35),
        Inches(1.35),
        Inches(12.6),
        Inches(4.2),
        hire,
        [Inches(2.4), Inches(1.8), Inches(5.6), Inches(2.8)],
    )
    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.35), Inches(5.8), Inches(12.6), Inches(1.3))
    _fill(box, NAVY)
    _textbox(
        s,
        Inches(0.55),
        Inches(6.05),
        Inches(12.2),
        Inches(0.9),
        "One-line recommendation: Adopt Drishti, BEH, ACRIMA, and RIM-ONE for GLAUCO external / adaptation.\n"
        "Do not fold them into the DR product. BIOMISA can wait.",
        size=15,
        bold=True,
        color=WHITE,
    )

    # ========== 7 SCRIPT ==========
    s = prs.slides.add_slide(blank)
    _bg(s)
    _banner(s, "5  ·  Speaking points (2–3 minutes)", "Say this analysis; pause on the bold numbers")
    points = [
        ("Disease", "All five are glaucoma (or glaucoma + macular). Not DR Acc packs → GLAUCO only."),
        ("Modality", "Four colour-fundus. BIOMISA = fundus + OCT but small OD cohort (~44 pts)."),
        ("India", "Only Drishti-GS1 is Indian — Aravind Madurai, 101 images with OD/cup GT."),
        ("Zero-shot", "AIROGS AUC 0.97 → Drishti 0.89–0.95 · BEH 0.78 · ACRIMA/RIM-ONE ~0.57 (OD-crop gap)."),
        ("Adaptation", "Fine-tune on ACRIMA → test AUC 0.997. Use ACRIMA for fine-tune; RIM-ONE as hard OOD."),
        ("Ask", "Confirm adoption list + keep GHOTA on APTOS/IDRiD/OEFI. OCTA-500/ROAD when access lands."),
    ]
    for i, (h, body) in enumerate(points):
        y = Inches(1.35 + i * 0.9)
        tag = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.45), y, Inches(2.0), Inches(0.7))
        _fill(tag, TEAL if i < 5 else NAVY)
        _textbox(s, Inches(0.45), y + Inches(0.18), Inches(2.0), Inches(0.4), h, size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(2.65), y, Inches(10.2), Inches(0.7))
        _fill(card, WHITE)
        card.line.color.rgb = RGBColor(0xD0, 0xDC, 0xE0)
        _textbox(s, Inches(2.85), y + Inches(0.18), Inches(9.8), Inches(0.45), body, size=13, color=INK)

    # ========== 8 NEXT / GET ==========
    s = prs.slides.add_slide(blank)
    _bg(s)
    _banner(s, "6  ·  Status — get / test / next", "Already on disk for this talk · optional OCTA later")
    cols = [
        (
            "Already have",
            GREEN,
            [
                "ACRIMA, BEH, RIM-ONE, Drishti, BIOMISA",
                "External catalogs built",
                "Zero-shot + ACRIMA FT results",
                "No IIIT register needed",
            ],
        ),
        (
            "Show / test",
            TEAL,
            [
                "This deck + Showcase 5b/5c",
                "External eval script (AIROGS → candidate sets)",
                "ACRIMA-FT → Drishti eval",
                "Memorize: 0.97 · 0.95 · 0.997 · 0.57",
            ],
        ),
        (
            "Optional next",
            AMBER,
            [
                "OCTA-500 / ROAD access",
                "Multi-site FT incl. RIM-ONE",
                "Standing Drishti/BEH externals",
                "Copy checkpoints to other PC",
            ],
        ),
    ]
    for i, (title, color, bullets) in enumerate(cols):
        x = Inches(0.4 + i * 4.25)
        card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(1.4), Inches(4.0), Inches(5.4))
        _fill(card, WHITE)
        card.line.color.rgb = color
        head = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, Inches(1.4), Inches(4.0), Inches(0.7))
        _fill(head, color)
        _textbox(s, x, Inches(1.55), Inches(4.0), Inches(0.45), title, size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        for j, b in enumerate(bullets):
            _textbox(s, x + Inches(0.25), Inches(2.4 + j * 0.9), Inches(3.5), Inches(0.7), "•  " + b, size=14, color=INK)

    # ========== 9 CLOSING ==========
    s = prs.slides.add_slide(blank)
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    _fill(bg, NAVY)
    stripe = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.2), Inches(7.5))
    _fill(stripe, TEAL)
    _textbox(s, Inches(0.8), Inches(2.0), Inches(11.5), Inches(0.5), "Decision we need", size=16, bold=True, color=TEAL_LT)
    _textbox(
        s,
        Inches(0.8),
        Inches(2.6),
        Inches(11.5),
        Inches(1.8),
        "Approve adoption of Drishti · BEH · ACRIMA · RIM-ONE\nfor GLAUCO external / adaptation — keep GHOTA on DR data only.",
        size=26,
        bold=True,
        color=WHITE,
    )
    _textbox(
        s,
        Inches(0.8),
        Inches(5.0),
        Inches(11.5),
        Inches(1.2),
        "Detail docs: GLAUCO_EXTERNAL_FUNDUS_BRIEF.md  ·  EXTERNAL_FUNDUS_RESULTS.md  ·  EXTERNAL_FUNDUS_SPECS.md",
        size=14,
        color=RGBColor(0xC8, 0xD8, 0xDE),
    )

    written = []
    for out in OUTS:
        out.parent.mkdir(parents=True, exist_ok=True)
        prs.save(str(out))
        written.append(out)
        print(f"Wrote {out} ({len(prs.slides)} slides)")
    return written


if __name__ == "__main__":
    build()
