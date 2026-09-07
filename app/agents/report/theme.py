"""
Report Theme

Central place for the "design tools" the Report Agent has access to —
fonts, colors, spacing. A static config rather than something the LLM
chooses per-report, so output stays consistent and fast to generate.
"""

from dataclasses import dataclass, field


@dataclass
class ReportTheme:
    # Fonts (ReportLab built-ins by default; swap for registered TTF names
    # via pdfmetrics.registerFont if custom fonts are added later)
    heading_font: str = "Helvetica-Bold"
    body_font: str = "Helvetica"

    # Core palette (hex)
    primary_color: str = "#1a1a2e"     # titles, headings
    accent_color: str = "#0f8b8d"      # highlights, table headers, chart bars
    text_color: str = "#2b2b2b"        # body text
    muted_color: str = "#6b6b6b"       # captions, sources, disclaimers

    # Chart-specific palette — a small gradient for ranked bars
    chart_palette: list[str] = field(default_factory=lambda: [
        "#0f8b8d", "#3fa9a5", "#6fc7bd", "#9fe5d5", "#cffbec",
    ])

    # Sizes
    title_size: int = 20
    heading_size: int = 14
    body_size: int = 10
    caption_size: int = 8

    # Layout
    page_margin: float = 54  # points (0.75in)