"""
Analyst Tool 5: Search / Retrieve More Evidence

Lets the Analyst Agent request additional evidence from the Retriever
when what it already has is insufficient to answer the question. This
is the mechanism behind the Analyst<->Retriever feedback loop described
in the spec — the Analyst never fabricates missing information, it asks
for more.
"""

import logging

from app.agents.retriever.retriever_agent import retrieve_evidence
from app.ingestion.document import Document

logger = logging.getLogger(__name__)


def request_more_evidence(
    follow_up_query: str,
    existing_evidence: list[Document],
    history: str = "",
    final_k: int = 4,
) -> list[Document]:
    """
    Runs a fresh, more specific query through the full Retriever pipeline
    and merges the results into the existing evidence bundle, skipping
    chunks already present (by content fingerprint, same dedupe strategy
    the Retriever itself uses).
    """
    if not follow_up_query or not follow_up_query.split():
        raise ValueError("Follow-up query must be a non-empty string.")

    logger.info("analyst requesting more evidence: %r", follow_up_query)

    new_evidence = retrieve_evidence(follow_up_query, history=history, final_k=final_k)

    seen = {doc.content[:150] for doc in existing_evidence}
    merged = list(existing_evidence)

    added = 0
    for doc in new_evidence:
        fingerprint = doc.content[:150]
        if fingerprint not in seen:
            seen.add(fingerprint)
            merged.append(doc)
            added += 1

    logger.info("merged %d new chunks (of %d returned) into evidence bundle", added, len(new_evidence))
    return merged