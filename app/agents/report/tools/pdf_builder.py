"""
Report Tool: PDF Builder

Low-level ReportLab primitives — themed headings, paragraphs, and
tables — kept separate from report_agent.py so the theme-application
logic (colors, fonts, sizing) lives in one place and stays consistent
across every section of the report.
"""

from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from app.agents.report.theme import ReportTheme


def build_document(out_path: str, theme: ReportTheme) -> SimpleDocTemplate:
    return SimpleDocTemplate(
        out_path,
        pagesize=letter,
        leftMargin=theme.page_margin,
        rightMargin=theme.page_margin,
        topMargin=theme.page_margin,
        bottomMargin=theme.page_margin,
    )


def title_style(theme: ReportTheme) -> ParagraphStyle:
    return ParagraphStyle(
        "ReportTitle",
        fontName=theme.heading_font,
        fontSize=theme.title_size,
        textColor=rl_colors.HexColor(theme.primary_color),
        spaceAfter=14,
    )


def heading_style(theme: ReportTheme) -> ParagraphStyle:
    return ParagraphStyle(
        "ReportHeading",
        fontName=theme.heading_font,
        fontSize=theme.heading_size,
        textColor=rl_colors.HexColor(theme.accent_color),
        spaceBefore=16,
        spaceAfter=8,
    )


def body_style(theme: ReportTheme) -> ParagraphStyle:
    return ParagraphStyle(
        "ReportBody",
        fontName=theme.body_font,
        fontSize=theme.body_size,
        textColor=rl_colors.HexColor(theme.text_color),
        leading=14,
        spaceAfter=8,
    )


def caption_style(theme: ReportTheme) -> ParagraphStyle:
    return ParagraphStyle(
        "ReportCaption",
        fontName=theme.body_font,
        fontSize=theme.caption_size,
        textColor=rl_colors.HexColor(theme.muted_color),
        leading=11,
    )


def build_table_flowable(headers: list[str], rows: list[list[str]], theme: ReportTheme) -> Table:
    data = [headers] + rows
    table = Table(data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor(theme.accent_color)),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), theme.heading_font),
        ("FONTNAME", (0, 1), (-1, -1), theme.body_font),
        ("FONTSIZE", (0, 0), (-1, -1), theme.body_size - 1),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor(theme.muted_color)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor("#f5f5f5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def build_chart_image(image_path: str, max_width: float = 5.5 * inch) -> Image:
    img = Image(image_path)
    aspect = img.imageHeight / img.imageWidth
    img.drawWidth = max_width
    img.drawHeight = max_width * aspect
    return img