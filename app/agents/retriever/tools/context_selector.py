"""
Tool 6: Context Selector

After reranking, trims the reranked list down to a compact, diverse
final set for the Analyst Agent — avoiding near-duplicate chunks and
capping total context size, rather than just taking a blind top-N.
"""

from app.ingestion.document import Document

def select_context(
    ranked_docs : list[Document],
    max_chunks : int = 6,
    max_chars : int = 6000,
    min_rerank_score: float = 4.0,
)-> list[Document]:
    """Greedily selects chunks in rank order, skipping near-duplicates
        (same source + overlapping content) and stopping once either the
        chunk-count or character budget is hit."""

    selected : list[Document] = []
    seen_sources : dict[str,set[str]] = {}
    total_chars = 0

    for doc in ranked_docs:
        if len(selected) >= max_chunks or total_chars >= max_chars:
            break

        score = doc.metadata.get("rerank_score")
        if score is not None and score < min_rerank_score:
            break

        source = doc.metadata.get("source_name", "unknown")
        content_key = doc.content[:100]  # Use first 100 chars as a simple content signature

        if source  in seen_sources and content_key in seen_sources[source]:
            continue  # Skip near-duplicate

        seen_sources.setdefault(source, set()).add(content_key)
        selected.append(doc)
        total_chars += len(doc.content)

    return selected
