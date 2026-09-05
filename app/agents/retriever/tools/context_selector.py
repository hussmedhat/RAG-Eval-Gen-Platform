"""
Tool 6: Context Selector

After reranking, trims the reranked list down to a compact, diverse
final set for the Analyst Agent — avoiding near-duplicate chunks and
capping total context size, rather than just taking a blind top-N.
"""

from app.ingestion.document import Document

def select_context(
    ranked_docs: list[Document],
    max_chunks: int = 6,
    max_chars: int = 6000,
    min_rerank_score: float = 4.0,
) -> list[Document]:
    """Greedily selects chunks in rank order, skipping near-duplicates
    and unscored (failed) candidates, and stopping once a genuinely
    low-relevance score is hit or a budget is exhausted."""

    selected: list[Document] = []
    seen_sources: dict[str, set[str]] = {}
    total_chars = 0

    for doc in ranked_docs:
        if len(selected) >= max_chunks or total_chars >= max_chars:
            break

        score = doc.metadata.get("rerank_score")

        if score is None:
            # Reranker call failed for this doc — don't treat as "confirmed irrelevant",
            # just skip it and keep scanning the rest of the ranked list.
            continue

        if score < min_rerank_score:
            # Genuinely low relevance, and list is sorted descending
            break

        source = doc.metadata.get("source_name", "unknown")
        content_key = doc.content[:100]

        if source in seen_sources and content_key in seen_sources[source]:
            continue

        seen_sources.setdefault(source, set()).add(content_key)
        selected.append(doc)
        total_chars += len(doc.content)

    return selected
