"""
Answer Tool: Response Formatter

Prepares the Analyst's structured findings (extracted tables,
numeric rankings) as markdown the LLM can weave directly into its
answer, and assembles the final packaged response (answer text +
sources) returned to the caller.
"""

from app.agents.analyst.tools.table_extractor import ExtractedTable


def render_table_markdown(table: ExtractedTable) -> str:
    """Renders an ExtractedTable as a GitHub-flavored markdown table."""
    if not table.headers or not table.rows:
        return ""
    header_row = "| " + " | ".join(table.headers) + " |"
    sep_row = "| " + " | ".join("---" for _ in table.headers) + " |"
    body_rows = "\n".join(
        "| " + " | ".join(str(cell) for cell in row) + " |"
        for row in table.rows
    )
    return "\n".join([header_row, sep_row, body_rows])


def render_ranking_markdown(data_analysis: dict | None) -> str:
    """Renders the Analyst's numeric ranking (if any) as a small
    markdown table — label, value — plus a one-line stats summary."""
    if not data_analysis or not data_analysis.get("ranking"):
        return ""

    ranking = data_analysis["ranking"]
    stats = data_analysis.get("stats", {})

    lines = ["| Rank | Label | Value |", "| --- | --- | --- |"]
    for i, point in enumerate(ranking, start=1):
        lines.append(f"| {i} | {point['label']} | {point['value']:.4g} |")

    table_md = "\n".join(lines)

    if stats:
        summary = (
            f"_mean: {stats.get('mean', 0):.4g}, "
            f"median: {stats.get('median', 0):.4g}"
            + (f", stdev: {stats['stdev']:.4g}" if stats.get("stdev") is not None else "")
            + "_"
        )
        return f"{table_md}\n\n{summary}"

    return table_md


def build_reference_context(tables: list[ExtractedTable], data_analysis: dict | None) -> str:
    """Bundles every markdown-rendered table/ranking the LLM has
    available into one block, so the prompt can point the model at
    exact numbers rather than letting it reconstruct them from memory."""
    blocks = []
    for i, table in enumerate(tables, start=1):
        md = render_table_markdown(table)
        if md:
            blocks.append(f"[Extracted table {i}]\n{md}")

    ranking_md = render_ranking_markdown(data_analysis)
    if ranking_md:
        blocks.append(f"[Data analysis ranking]\n{ranking_md}")

    return "\n\n".join(blocks) if blocks else "(no structured tables extracted)"


def assemble_response(answer_text: str, sources_block: str) -> str:
    """Final packaging: answer body followed by the sources list."""
    if not sources_block:
        return answer_text
    return f"{answer_text}\n\n{sources_block}"