"""
Report Agent

Formats an already-produced AgenticWorkflowResult (answer text,
citations, extracted tables, comparison, numeric ranking) into a
styled PDF. Does not call an LLM and does not re-judge anything — the
Retriever/Analyst/Answer agents already decided what's true and
sufficient; this agent's only job is presentation.
"""

import logging
import re
import tempfile
from pathlib import Path

from reportlab.platypus import Spacer

from app.agents.report.theme import ReportTheme
from app.agents.report.tools.chart_renderer import render_ranking_chart
from app.agents.report.tools.pdf_builder import (
    body_style, build_chart_image, build_document, build_table_flowable,
    caption_style, heading_style, title_style,
)

logger = logging.getLogger(__name__)

_CITATION_MARKER = re.compile(r"\[(\d+)\]")


def _clean_answer_text(answer: str) -> str:
    """The workflow's `answer` field already has the sources block
    appended (assemble_response in response_formatter.py). Split it back
    off here since the PDF renders sources as its own styled section."""
    if "**Sources:**" in answer:
        return answer.split("**Sources:**")[0].strip()
    return answer.strip()


def _extract_sources_block(answer: str) -> str | None:
    if "**Sources:**" not in answer:
        return None
    return answer.split("**Sources:**", 1)[1].strip()


def generate_report(
    question: str,
    workflow_result,  # AgenticWorkflowResult from agentic_workflow.py
    theme: ReportTheme | None = None,
    out_dir: str | None = None,
) -> str:
    """Builds a themed PDF from an already-completed workflow result.
    Returns the path to the generated PDF file."""
    theme = theme or ReportTheme()
    out_dir_path = Path(out_dir) if out_dir else Path(tempfile.gettempdir())
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir_path / f"report_{abs(hash(question))}.pdf")

    doc = build_document(out_path, theme)
    story = []

    # --- Title + question ---
    story.append(Paragraph_safe(question, title_style(theme)))
    story.append(Spacer(1, 6))

    # --- Answer ---
    story.append(Paragraph_safe("Answer", heading_style(theme)))
    answer_text = _clean_answer_text(workflow_result.answer)
    story.append(Paragraph_safe(answer_text, body_style(theme)))

    if workflow_result.disclaimer:
        story.append(Spacer(1, 6))
        story.append(Paragraph_safe(f"⚠ {workflow_result.disclaimer}", caption_style(theme)))

    # --- Key facts ---
    if workflow_result.key_facts:
        story.append(Paragraph_safe("Key Facts", heading_style(theme)))
        for fact in workflow_result.key_facts:
            story.append(Paragraph_safe(f"• {fact}", body_style(theme)))

    # --- Extracted tables ---
    for i, table in enumerate(workflow_result.tables, start=1):
        headers, rows = table.get("headers"), table.get("rows")
        if headers and rows:
            story.append(Paragraph_safe(f"Extracted Table {i}", heading_style(theme)))
            story.append(build_table_flowable(headers, rows, theme))
            story.append(Spacer(1, 10))

    # --- Numeric ranking: table + chart ---
    if workflow_result.data_analysis and workflow_result.data_analysis.get("ranking"):
        story.append(Paragraph_safe("Comparison / Ranking", heading_style(theme)))

        ranking = workflow_result.data_analysis["ranking"]
        rows = [[point["label"], f"{point['value']:.4g}"] for point in ranking]
        story.append(build_table_flowable(["Label", "Value"], rows, theme))
        story.append(Spacer(1, 10))

        chart_path = render_ranking_chart(workflow_result.data_analysis, theme)
        if chart_path:
            story.append(build_chart_image(chart_path))
            story.append(Spacer(1, 10))

    # --- Comparison across sources ---
    comparison = workflow_result.comparison
    if comparison:
        story.append(Paragraph_safe("Cross-Source Comparison", heading_style(theme)))
        if comparison.get("agreements"):
            story.append(Paragraph_safe("Agreements:", body_style(theme)))
            for point in comparison["agreements"]:
                story.append(Paragraph_safe(f"• {point}", body_style(theme)))
        if comparison.get("contradictions"):
            story.append(Paragraph_safe("Contradictions:", body_style(theme)))
            for point in comparison["contradictions"]:
                story.append(Paragraph_safe(f"• {point}", body_style(theme)))

    # --- Sources ---
    sources_block = _extract_sources_block(workflow_result.answer)
    if sources_block:
        story.append(Paragraph_safe("Sources", heading_style(theme)))
        for line in sources_block.split("\n"):
            if line.strip():
                story.append(Paragraph_safe(line.strip(), caption_style(theme)))

    doc.build(story)
    logger.info("generated report PDF at %s", out_path)
    return out_path


def Paragraph_safe(text: str, style):
    """ReportLab's Paragraph treats text as mini-HTML, so raw text needs
    escaping to avoid '<' or '&' in real content breaking the parser."""
    from xml.sax.saxutils import escape
    from reportlab.platypus import Paragraph
    return Paragraph(escape(text).replace("\n", "<br/>"), style)