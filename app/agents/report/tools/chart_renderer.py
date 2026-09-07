"""
Report Tool: Chart Renderer

Renders the Analyst's numeric ranking (data_analysis["ranking"]) as a
horizontal bar chart image, themed to match the rest of the report.
Returns None if there's nothing to chart — a report without numeric
comparison data just skips the chart section entirely.
"""

import logging
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display backend needed on a server
import matplotlib.pyplot as plt

from app.agents.report.theme import ReportTheme

logger = logging.getLogger(__name__)


def render_ranking_chart(data_analysis: dict | None, theme: ReportTheme) -> str | None:
    """Renders data_analysis['ranking'] (list of {'label', 'value'} dicts,
    already sorted) as a horizontal bar chart PNG. Returns the file path,
    or None if there's no ranking data to chart."""
    if not data_analysis or not data_analysis.get("ranking"):
        return None

    ranking = data_analysis["ranking"]
    labels = [point["label"] for point in ranking]
    values = [point["value"] for point in ranking]

    # Chart reads top-to-bottom in rank order, so reverse for barh
    # (matplotlib draws barh bottom-to-top)
    labels = labels[::-1]
    values = values[::-1]

    colors = [
        theme.chart_palette[i % len(theme.chart_palette)]
        for i in range(len(values))
    ][::-1]

    fig, ax = plt.subplots(figsize=(6, 0.5 * len(values) + 1), dpi=150)
    bars = ax.barh(labels, values, color=colors)

    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(theme.muted_color)
    ax.spines["bottom"].set_color(theme.muted_color)
    ax.tick_params(colors=theme.text_color, labelsize=9)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3g}",
            va="center", fontsize=8, color=theme.text_color,
        )

    fig.tight_layout()

    out_path = Path(tempfile.gettempdir()) / f"report_chart_{id(data_analysis)}.png"
    fig.savefig(out_path, transparent=False)
    plt.close(fig)

    logger.info("rendered ranking chart with %d bars to %s", len(values), out_path)
    return str(out_path)