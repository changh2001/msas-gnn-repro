#!/usr/bin/env python3
r"""
Post-process DOCX layout for thesis-like formatting.

This script edits the Word file directly (python-docx), mainly:
- normalize Normal paragraphs (indent/spacing/alignment)
- map probable figure/table caption lines to caption styles
- tune heading paragraph spacing
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Pt


FIG_PATTERNS = [
    re.compile(r"^图\s*\d"),
    re.compile(r"示意图$"),
    re.compile(r"框架图$"),
    re.compile(r"流程图$"),
    re.compile(r"关系图$"),
]

TAB_PATTERNS = [
    re.compile(r"^表\s*\d"),
    re.compile(r"统计信息$"),
    re.compile(r"结果对比$"),
]


def looks_like_figure_caption(text: str) -> bool:
    t = text.strip()
    return any(p.search(t) for p in FIG_PATTERNS)


def looks_like_table_caption(text: str) -> bool:
    t = text.strip()
    return any(p.search(t) for p in TAB_PATTERNS)


def set_east_asia_font(paragraph, font_name: str) -> None:
    for run in paragraph.runs:
        run.font.name = font_name
        r = run._element
        rPr = r.get_or_add_rPr()
        rFonts = rPr.get_or_add_rFonts()
        rFonts.set(qn("w:eastAsia"), font_name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-fix thesis DOCX layout.")
    parser.add_argument("--infile", required=True)
    parser.add_argument("--outfile", required=True)
    args = parser.parse_args()

    doc = Document(args.infile)
    style_names = {s.name for s in doc.styles}

    fig_style = "图标题格式" if "图标题格式" in style_names else "Caption"
    tab_style = "表标题格式" if "表标题格式" in style_names else "Caption"

    for p in doc.paragraphs:
        text = p.text.strip()
        style = p.style.name if p.style is not None else ""

        # Caption heuristic.
        if text and style in {"Normal", "Body Text", "Body Text Indent", "No Spacing"}:
            if looks_like_figure_caption(text):
                p.style = doc.styles[fig_style]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                continue
            if looks_like_table_caption(text):
                p.style = doc.styles[tab_style]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                continue

        # Heading spacing polish.
        if style.startswith("Heading"):
            pf = p.paragraph_format
            pf.space_before = Pt(12)
            pf.space_after = Pt(6)
            pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
            if style == "Heading 1":
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            continue

        # Normal paragraph normalization.
        if style == "Normal":
            pf = p.paragraph_format
            pf.first_line_indent = Pt(24)  # ~2 Chinese chars in common setup
            pf.line_spacing = 1.5
            pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
            pf.space_before = Pt(0)
            pf.space_after = Pt(0)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            set_east_asia_font(p, "宋体")

    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
