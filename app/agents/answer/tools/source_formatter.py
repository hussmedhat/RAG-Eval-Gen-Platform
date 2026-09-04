"""
Answer Tool: Source Formatter

Builds the final "Sources" reference list that accompanies the
answer, matching the sequential [n] markers produced by the Citation
Formatter. Each entry names the source document and, where available,
the page/section/slide the cited chunk came from — the "precise page
and document citations" called for in the spec.
"""

from app.ingestion.document import Document


def _locate(doc: Document) -> str | None:
    """Best-effort human-readable locator for a chunk, e.g. 'p. 4' or
    'slide 3' — whichever metadata the source's loader attached.
    Returns None if nothing locator-like exists."""
    meta = doc.metadata
    if "page" in meta:
        return f"p. {meta['page']}"
    if "slide_number" in meta:
        return f"slide {meta['slide_number']}"
    if "section" in meta:
        return f"section: {meta['section']}"
    if "table_index" in meta:
        return f"table {meta['table_index']}"
    if "paragraph_index" in meta:
        return f"¶{meta['paragraph_index']}"
    if "timestamp_start" in meta:
        return f"{meta['timestamp_start']:.0f}s"
    return None


def format_sources(cited_evidence: list[Document]) -> str:
    """Renders the numbered source list matching citation markers
    [1..N] (cited_evidence is already in that order)."""
    if not cited_evidence:
        return ""

    lines = ["**Sources:**"]
    for i, doc in enumerate(cited_evidence, start=1):
        name = doc.metadata.get("source_name", "unknown source")
        locator = _locate(doc)
        entry = f"{i}. {name}" + (f", {locator}" if locator else "")
        lines.append(entry)

    return "\n".join(lines)