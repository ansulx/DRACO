#!/usr/bin/env python3
"""Build GLAUCO supervisor PPT — story arc: disease → datasets → classes → problem → methods → results/graphs."""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

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
    accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.08), Inches(7.5))
    _fill(accent, TEAL)


def _banner(slide, title: str, subtitle: str):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.05))
    _fill(bar, NAVY)
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


def _kpi(slide, left, top, value, label, *, accent=TEAL, w=2.0, h=1.2):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, Inches(w), Inches(h))
    _fill(card, WHITE)
    card.line.color.rgb = accent
    card.line.width = Pt(1.25)
    _textbox(slide, left, top + Inches(0.18), Inches(w), Inches(0.5), value, size=22, bold=True, color=accent, align=PP_ALIGN.CENTER)
    _textbox(slide, left, top + Inches(0.7), Inches(w), Inches(0.4), label, size=11, color=MUTED, align=PP_ALIGN.CENTER)


def _style_chart(chart, *, max_val=1.0):
    chart.has_legend = False
    plot = chart.plots[0]
    plot.has_data_labels = True
    plot.data_labels.font.size = Pt(10)
    plot.data_labels.font.bold = True
    plot.data_labels.number_format = "0.00"
    try:
        chart.value_axis.maximum_scale = max_val
        chart.value_axis.minimum_scale = 0
        chart.value_axis.has_major_gridlines = True
    except Exception:
        pass
    try:
        series = chart.series[0]
        series.format.fill.solid()
        series.format.fill.fore_color.rgb = TEAL
    except Exception:
        pass


def build() -> list[Path]:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # ========== 1 TITLE ==========
    s = prs.slides.add_slide(blank)
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    _fill(bg, NAVY)
    panel = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(5.6), Inches(13.333), Inches(1.9))
    _fill(panel, TEAL)
    _textbox(s, Inches(0.7), Inches(1.8), Inches(11.5), Inches(0.5), "GLAUCO  ·  Glaucoma screening only", size=16, bold=True, color=TEAL_LT)
    _textbox(s, Inches(0.7), Inches(2.35), Inches(11.5), Inches(1.0), "From datasets → methods → metrics", size=32, bold=True, color=WHITE)
    _textbox(
        s,
        Inches(0.7),
        Inches(3.5),
        Inches(11.5),
        Inches(0.8),
        "Disease · datasets · classes · problem · methodology · results (sens / spec / F1 / AUC)\nPeak packs for edge handoff",
        size=15,
        color=RGBColor(0xC8, 0xD8, 0xDE),
    )
    _textbox(s, Inches(0.7), Inches(6.05), Inches(11.5), Inches(0.9), "Supervisor briefing  ·  Speak in this order", size=14, bold=True, color=WHITE)

    # ========== 2 AGENDA / HOW TO START ==========
    s = prs.slides.add_slide(blank)
    _bg(s)
    _banner(s, "How to start — talk order", "Say each block once · then land on graphs")
    items = [
        ("01", "Disease", "This is glaucoma — GLAUCO only (not DR / GHOTA)"),
        ("02", "Datasets", "What we trained on · what we only tested"),
        ("03", "Classes", "Peak = 2-class screening · GLEAM = 4-stage"),
        ("04", "Problem", "Screen normal vs glaucoma across fundus + OCT / RNFLT"),
        ("05", "Methodology", "Train specialists · freeze peak · external OOD tests"),
        ("06", "Results + graphs", "Sens / Spec / Prec / Rec / F1 / Acc / AUC — then bars"),
    ]
    for i, (num, title, body) in enumerate(items):
        col = i % 2
        row = i // 2
        x = Inches(0.4 + col * 6.45)
        y = Inches(1.35 + row * 1.85)
        chip = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, Inches(1.0), Inches(1.5))
        _fill(chip, NAVY if i < 5 else TEAL)
        _textbox(s, x, y + Inches(0.5), Inches(1.0), Inches(0.5), num, size=20, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x + Inches(1.15), y, Inches(5.1), Inches(1.5))
        _fill(card, WHITE)
        card.line.color.rgb = TEAL_LT
        _textbox(s, x + Inches(1.35), y + Inches(0.3), Inches(4.7), Inches(0.4), title, size=18, bold=True, color=NAVY)
        _textbox(s, x + Inches(1.35), y + Inches(0.8), Inches(4.7), Inches(0.5), body, size=13, color=MUTED)

    # ========== 3 DISEASE ==========
    s = prs.slides.add_slide(blank)
    _bg(s)
    _banner(s, "1  ·  Disease — glaucoma", "Why this product exists")
    left = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.4), Inches(1.4), Inches(6.1), Inches(5.5))
    _fill(left, WHITE)
    left.line.color.rgb = TEAL
    _textbox(s, Inches(0.7), Inches(1.65), Inches(5.6), Inches(0.4), "Clinical framing", size=16, bold=True, color=TEAL)
    for i, line in enumerate(
        [
            "Glaucoma damages the optic nerve — often silent until late.",
            "Screening needs high sensitivity without drowning clinics in false alarms (specificity).",
            "Inputs we can automate: colour fundus photos + OCT / RNFLT maps.",
            "Outcome for edge: binary score — normal vs glaucoma.",
        ]
    ):
        _textbox(s, Inches(0.7), Inches(2.3 + i * 1.0), Inches(5.6), Inches(0.85), "▸  " + line, size=14, color=INK)

    right = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.4), Inches(6.1), Inches(5.5))
    _fill(right, WHITE)
    right.line.color.rgb = NAVY
    _textbox(s, Inches(7.1), Inches(1.65), Inches(5.6), Inches(0.4), "Product boundary", size=16, bold=True, color=NAVY)
    for i, line in enumerate(
        [
            "GLAUCO = glaucoma only.",
            "GHOTA = diabetic retinopathy (separate).",
            "These Downloads / Kaggle sets are glaucoma — they do not go into GHOTA.",
            "Indian DR story stays on IDRiD / APTOS under GHOTA.",
        ]
    ):
        _textbox(s, Inches(7.1), Inches(2.3 + i * 1.0), Inches(5.6), Inches(0.85), "▸  " + line, size=14, color=INK)

    # ========== 4 DATASETS ==========
    s = prs.slides.add_slide(blank)
    _bg(s)
    _banner(s, "2  ·  Datasets we used", "Train = learns weights  ·  External = test only (no Acc leak)")
    fund = [
        ["Dataset", "Modality", "Role", "In peak .pth?"],
        ["AIROGS-light", "Fundus", "PRIMARY fundus train", "YES — fundus peak"],
        ["Harvard-GD", "RNFLT / OCT", "PRIMARY OCT train", "YES — OCT peak"],
        ["ACRIMA (Downloads)", "Fundus OD-crop", "Fine-tune adaptation", "Separate FT ckpt"],
        ["Harvard-GDP / GF", "RNFLT", "Scale / bench", "No (scale)"],
        ["GLEAM", "Fundus + OCT (+VF)", "Staging / fusion research", "No (4-class track)"],
        ["Drishti · BEH · RIM-ONE", "Fundus", "EXTERNAL test only", "No — never train"],
        ["BIOMISA", "Fundus + OCT", "Cataloged; later", "No"],
    ]
    _table(
        s,
        Inches(0.35),
        Inches(1.3),
        Inches(12.6),
        Inches(5.0),
        fund,
        [Inches(3.4), Inches(2.4), Inches(3.6), Inches(3.2)],
    )
    _textbox(
        s,
        Inches(0.4),
        Inches(6.5),
        Inches(12.5),
        Inches(0.6),
        "Say aloud: “Peak fundus Acc is AIROGS. Peak OCT Acc is Harvard-GD. Drishti/BEH/RIM-ONE prove transfer — they never trained the peak Acc.”",
        size=13,
        bold=True,
        color=MUTED,
    )

    # ========== 5 CLASSES ==========
    s = prs.slides.add_slide(blank)
    _bg(s)
    _banner(s, "3  ·  Classes", "What the model outputs")
    cls_rows = [
        ["Setting", "Classes", "Labels", "Where"],
        ["Peak fundus pack (edge)", "2", "normal / glaucoma", "glauco_fundus_peak.pth"],
        ["Peak OCT RNFLT (edge)", "2", "normal / glaucoma", "glauco_oct_rnflt_edge.pth"],
        ["GLEAM staging / fusion", "4", "normal → early → mid → adv", "Research track"],
        ["3-class option", "3", "normal / early / advanced", "If supervisors require staging"],
    ]
    _table(
        s,
        Inches(0.35),
        Inches(1.4),
        Inches(12.6),
        Inches(3.2),
        cls_rows,
        [Inches(3.4), Inches(1.4), Inches(4.0), Inches(3.8)],
    )
    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.35), Inches(5.0), Inches(12.6), Inches(2.0))
    _fill(box, WHITE)
    box.line.color.rgb = TEAL
    _textbox(s, Inches(0.55), Inches(5.2), Inches(12.2), Inches(0.4), "Defense line", size=14, bold=True, color=TEAL)
    _textbox(
        s,
        Inches(0.55),
        Inches(5.7),
        Inches(12.2),
        Inches(1.1),
        "Deploy pack is binary screening (glaucoma vs normal) — clinical Acc target for edge.\n"
        "We also trained 4-class staging on GLEAM. Three-class is possible; binary is what we froze for the .pth handoff.",
        size=14,
        color=INK,
    )

    # ========== 6 PROBLEM ==========
    s = prs.slides.add_slide(blank)
    _bg(s)
    _banner(s, "4  ·  Problem", "What we set out to solve")
    probs = [
        ("Screening goal", "Detect glaucoma from fundus and/or OCT-derived RNFLT with measurable Sens / Spec."),
        ("Train vs test honesty", "Primary Acc on AIROGS (fundus) + Harvard-GD (OCT). Externals never touch peak Acc."),
        ("Domain shift", "OD-crop sets (ACRIMA / RIM-ONE) break full-field models — need adaptation, not fake zero-shot."),
        ("Edge handoff", "Ship .pth + architecture + params + memory + full metric card for the deploy team."),
    ]
    for i, (h, body) in enumerate(probs):
        y = Inches(1.35 + i * 1.35)
        tag = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.45), y, Inches(2.6), Inches(1.1))
        _fill(tag, TEAL if i % 2 == 0 else NAVY)
        _textbox(s, Inches(0.55), y + Inches(0.35), Inches(2.4), Inches(0.5), h, size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(3.25), y, Inches(9.6), Inches(1.1))
        _fill(card, WHITE)
        card.line.color.rgb = RGBColor(0xD0, 0xDC, 0xE0)
        _textbox(s, Inches(3.5), y + Inches(0.35), Inches(9.2), Inches(0.5), body, size=15, color=INK)

    # ========== 7 METHODOLOGY ==========
    s = prs.slides.add_slide(blank)
    _bg(s)
    _banner(s, "5  ·  Methodology", "How we built and tested")
    steps = [
        ("A", "Catalog", "Normalize Downloads / Kaggle into GLAUCO CSVs (path + label)."),
        ("B", "Train fundus", "RETFound-CFP on AIROGS-light → peak fundus Acc."),
        ("C", "Train OCT", "EfficientNet-B0 on Harvard-GD RNFLT → edge OCT Acc."),
        ("D", "Adapt", "Fine-tune ACRIMA for OD-crop domain (separate ckpt)."),
        ("E", "External eval", "Zero-shot on Drishti / BEH / ACRIMA / RIM-ONE — report Sens/Spec/AUC."),
        ("F", "Export", "Freeze glauco_edge_v1 .pth + model card (params, memory, metrics)."),
    ]
    for i, (num, title, body) in enumerate(steps):
        col = i % 3
        row = i // 3
        x = Inches(0.4 + col * 4.25)
        y = Inches(1.4 + row * 2.7)
        card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, Inches(4.0), Inches(2.4))
        _fill(card, WHITE)
        card.line.color.rgb = TEAL
        chip = s.shapes.add_shape(MSO_SHAPE.OVAL, x + Inches(0.25), y + Inches(0.3), Inches(0.7), Inches(0.7))
        _fill(chip, NAVY)
        _textbox(s, x + Inches(0.25), y + Inches(0.42), Inches(0.7), Inches(0.45), num, size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        _textbox(s, x + Inches(1.1), y + Inches(0.4), Inches(2.7), Inches(0.45), title, size=16, bold=True, color=NAVY)
        _textbox(s, x + Inches(0.3), y + Inches(1.2), Inches(3.4), Inches(0.95), body, size=13, color=INK)

    # ========== 8 RESULTS — METRIC CARDS ==========
    s = prs.slides.add_slide(blank)
    _bg(s)
    _banner(s, "6  ·  Results — peak metrics", "This is the Sens / Spec / Prec / Rec / F1 / Acc / AUC block")

    _textbox(s, Inches(0.4), Inches(1.25), Inches(12.5), Inches(0.35), "Fundus peak · AIROGS holdout (glauco_fundus_peak.pth)", size=14, bold=True, color=NAVY)
    metrics_f = [
        ("0.920", "Accuracy", TEAL),
        ("0.920", "Precision", TEAL),
        ("0.920", "Recall / Sens", GREEN),
        ("0.920", "Specificity", GREEN),
        ("0.920", "F1", TEAL),
        ("0.969", "AUC", NAVY),
    ]
    for i, (val, lab, acc) in enumerate(metrics_f):
        _kpi(s, Inches(0.35 + i * 2.15), Inches(1.65), val, lab, accent=acc, w=2.0, h=1.15)

    _textbox(s, Inches(0.4), Inches(3.05), Inches(12.5), Inches(0.35), "OCT peak · Harvard-GD test (glauco_oct_rnflt_edge.pth)", size=14, bold=True, color=NAVY)
    metrics_o = [
        ("0.870", "Accuracy", TEAL),
        ("0.865", "Precision", TEAL),
        ("0.882", "Recall / Sens", GREEN),
        ("0.857", "Specificity", GREEN),
        ("0.874", "F1", TEAL),
        ("0.906", "AUC", NAVY),
    ]
    for i, (val, lab, acc) in enumerate(metrics_o):
        _kpi(s, Inches(0.35 + i * 2.15), Inches(3.45), val, lab, accent=acc, w=2.0, h=1.15)

    note = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.35), Inches(4.9), Inches(12.6), Inches(2.1))
    _fill(note, WHITE)
    note.line.color.rgb = TEAL
    _textbox(s, Inches(0.55), Inches(5.1), Inches(12.2), Inches(0.35), "How to say it", size=14, bold=True, color=TEAL)
    _textbox(
        s,
        Inches(0.55),
        Inches(5.55),
        Inches(12.2),
        Inches(1.2),
        "Fundus: Acc / Sens / Spec / F1 all ≈ 0.92, AUC 0.97 on holdout — balanced binary screen.\n"
        "OCT edge: lighter model (4M params, ~15 MB) — Acc 0.87, Sens 0.88, Spec 0.86, AUC 0.91 (thr 0.55).\n"
        "Extra clinical knob: fundus Sens @ 95% Spec = 0.83 (holdout).",
        size=14,
        color=INK,
    )

    # ========== 9 GRAPHS — EXTERNAL AUC ==========
    s = prs.slides.add_slide(blank)
    _bg(s)
    _banner(s, "7  ·  Graphs — external fundus AUC", "Same peak fundus .pth · zero-shot (no retraining)")

    chart_data = CategoryChartData()
    chart_data.categories = ["AIROGS\nholdout", "Drishti\nIndia", "Drishti\nofficial", "BEH", "ACRIMA\nOD-crop", "RIM-ONE\nhospital"]
    chart_data.add_series("AUC", (0.969, 0.890, 0.951, 0.782, 0.578, 0.567))
    chart = s.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        Inches(0.4),
        Inches(1.3),
        Inches(8.2),
        Inches(5.5),
        chart_data,
    ).chart
    _style_chart(chart, max_val=1.05)

    side = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.9), Inches(1.4), Inches(4.0), Inches(5.4))
    _fill(side, WHITE)
    side.line.color.rgb = NAVY
    _textbox(s, Inches(9.1), Inches(1.65), Inches(3.6), Inches(0.4), "Read the graph", size=16, bold=True, color=NAVY)
    for i, line in enumerate(
        [
            "In-domain (AIROGS): ~0.97",
            "India (Drishti): 0.89–0.95 — transfers",
            "BEH site-shift: ~0.78 — usable",
            "OD-crop zero-shot: ~0.57 — expected fail",
            "Fix for OD-crop: ACRIMA FT → AUC 0.997",
            "RIM-ONE stays hard multi-site bench",
        ]
    ):
        _textbox(s, Inches(9.15), Inches(2.3 + i * 0.65), Inches(3.55), Inches(0.55), "•  " + line, size=12, color=INK)

    # ========== 10 GRAPHS — METRIC BARS FUNDUS vs OCT ==========
    s = prs.slides.add_slide(blank)
    _bg(s)
    _banner(s, "8  ·  Graphs — fundus vs OCT screening metrics", "Side-by-side on the six headline numbers")

    chart_data2 = CategoryChartData()
    chart_data2.categories = ["Acc", "Prec", "Sens", "Spec", "F1", "AUC"]
    chart_data2.add_series("Fundus (AIROGS)", (0.920, 0.920, 0.920, 0.920, 0.920, 0.969))
    chart_data2.add_series("OCT (Harvard-GD)", (0.870, 0.865, 0.882, 0.857, 0.874, 0.906))
    chart2 = s.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        Inches(0.4),
        Inches(1.3),
        Inches(12.5),
        Inches(5.0),
        chart_data2,
    ).chart
    chart2.has_legend = True
    chart2.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart2.legend.include_in_layout = False
    try:
        chart2.value_axis.maximum_scale = 1.05
        chart2.value_axis.minimum_scale = 0
    except Exception:
        pass
    try:
        chart2.series[0].format.fill.solid()
        chart2.series[0].format.fill.fore_color.rgb = TEAL
        chart2.series[1].format.fill.solid()
        chart2.series[1].format.fill.fore_color.rgb = NAVY
    except Exception:
        pass
    _textbox(
        s,
        Inches(0.5),
        Inches(6.5),
        Inches(12.3),
        Inches(0.6),
        "Fundus = accuracy peak (heavy). OCT = edge-friendly (light). Both binary · both ready as .pth packs.",
        size=13,
        bold=True,
        color=MUTED,
    )

    # ========== 11 EXTERNAL TABLE (SENS/SPEC DETAIL) ==========
    s = prs.slides.add_slide(blank)
    _bg(s)
    _banner(s, "9  ·  External table — Sens / Spec detail", "Zero-shot peak fundus model on Downloads / Kaggle sets")
    exp = [
        ["Dataset", "n", "AUC", "Acc", "Sens", "Spec", "Takeaway"],
        ["Drishti full (India)", "101", "0.890", "0.812", "0.786", "0.871", "Indian external — adopt"],
        ["Drishti official test", "51", "0.951", "0.863", "0.816", "1.000", "Strong on official split"],
        ["BEH", "634", "0.782", "0.793", "0.585", "0.870", "South-Asia site shift"],
        ["ACRIMA OD-crop", "705", "0.578", "0.452", "0.045", "0.974", "FT data — not zero-shot"],
        ["RIM-ONE hospital", "174", "0.567", "0.690", "0.071", "0.983", "Hard multi-site bench"],
    ]
    _table(
        s,
        Inches(0.25),
        Inches(1.35),
        Inches(12.8),
        Inches(4.2),
        exp,
        [Inches(2.5), Inches(0.9), Inches(1.1), Inches(1.1), Inches(1.1), Inches(1.1), Inches(5.0)],
    )
    _textbox(
        s,
        Inches(0.4),
        Inches(5.8),
        Inches(12.5),
        Inches(1.2),
        "After ACRIMA fine-tune: ACRIMA test AUC 0.997 · Acc 0.972 · Sens@95%Spec 0.987.\n"
        "Drishti / BEH stay strong after FT; RIM-ONE hospital OOD still hard — that is the stress test.",
        size=14,
        color=INK,
    )

    # ========== 12 PEAK PACKS ==========
    s = prs.slides.add_slide(blank)
    _bg(s)
    _banner(s, "10  ·  Peak packs for edge", "What we hand the deploy team")
    left = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.4), Inches(1.35), Inches(6.1), Inches(5.6))
    _fill(left, WHITE)
    left.line.color.rgb = TEAL
    head = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.4), Inches(1.35), Inches(6.1), Inches(0.65))
    _fill(head, TEAL)
    _textbox(s, Inches(0.4), Inches(1.48), Inches(6.1), Inches(0.4), "FUNDUS peak", size=18, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    for i, line in enumerate(
        [
            "File: glauco_fundus_peak.pth",
            "Train: AIROGS-light · 2 classes",
            "Arch: RETFound-CFP · 303 M params",
            "Weights ~1157 MB · peak CUDA ~2.3 GB",
            "Holdout: Acc 0.92 · AUC 0.97",
            "Full card: sens/spec/prec/rec/F1",
        ]
    ):
        _textbox(s, Inches(0.65), Inches(2.25 + i * 0.7), Inches(5.6), Inches(0.55), "▸  " + line, size=14, color=INK)

    right = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.35), Inches(6.1), Inches(5.6))
    _fill(right, WHITE)
    right.line.color.rgb = NAVY
    head2 = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(6.8), Inches(1.35), Inches(6.1), Inches(0.65))
    _fill(head2, NAVY)
    _textbox(s, Inches(6.8), Inches(1.48), Inches(6.1), Inches(0.4), "OCT peak (edge-friendly)", size=18, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    for i, line in enumerate(
        [
            "File: glauco_oct_rnflt_edge.pth",
            "Train: Harvard-GD · 2 classes",
            "Arch: EfficientNet-B0 · 4.0 M params",
            "Weights ~15.5 MB · peak CUDA ~43 MB",
            "Test: Acc 0.87 · Sens 0.88 · AUC 0.906",
            "Thr 0.55 · full sens/spec/F1 card",
        ]
    ):
        _textbox(s, Inches(7.05), Inches(2.25 + i * 0.7), Inches(5.6), Inches(0.55), "▸  " + line, size=14, color=INK)

    # ========== 13 SPEAKING SCRIPT ==========
    s = prs.slides.add_slide(blank)
    _bg(s)
    _banner(s, "11  ·  Speak this (2–3 min)", "One sentence per block — then open the graphs")
    points = [
        ("Start", "These are glaucoma datasets under GLAUCO — not DR."),
        ("Datasets", "Train: AIROGS fundus + Harvard-GD OCT. Externals: Drishti, BEH, ACRIMA, RIM-ONE."),
        ("Classes", "Peak packs are 2-class: normal vs glaucoma."),
        ("Problem", "Binary screening with honest train/test split + domain-shift awareness."),
        ("Method", "Specialists → external OOD → adapt OD-crop → export .pth + metric card."),
        ("Results", "Fundus AUC 0.97 · OCT Acc 0.87 / Sens 0.88 · Drishti 0.89–0.95 — then bars."),
    ]
    for i, (h, body) in enumerate(points):
        y = Inches(1.3 + i * 0.9)
        tag = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.45), y, Inches(2.0), Inches(0.7))
        _fill(tag, TEAL if i < 5 else NAVY)
        _textbox(s, Inches(0.45), y + Inches(0.18), Inches(2.0), Inches(0.4), h, size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(2.65), y, Inches(10.2), Inches(0.7))
        _fill(card, WHITE)
        card.line.color.rgb = RGBColor(0xD0, 0xDC, 0xE0)
        _textbox(s, Inches(2.85), y + Inches(0.18), Inches(9.8), Inches(0.45), body, size=13, color=INK)

    # ========== 14 CLOSE ==========
    s = prs.slides.add_slide(blank)
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    _fill(bg, NAVY)
    stripe = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.2), Inches(7.5))
    _fill(stripe, TEAL)
    _textbox(s, Inches(0.8), Inches(2.0), Inches(11.5), Inches(0.5), "Ask", size=16, bold=True, color=TEAL_LT)
    _textbox(
        s,
        Inches(0.8),
        Inches(2.6),
        Inches(11.5),
        Inches(1.8),
        "Confirm binary edge packs:\nfundus = AIROGS · OCT = Harvard-GD · externals stay test-only",
        size=24,
        bold=True,
        color=WHITE,
    )
    _textbox(
        s,
        Inches(0.8),
        Inches(5.0),
        Inches(11.5),
        Inches(1.2),
        "Detail: docs/GLAUCO_DATASETS_USED.md  ·  docs/GLAUCO_EDGE_METRICS.md  ·  exports/glauco_edge/",
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
