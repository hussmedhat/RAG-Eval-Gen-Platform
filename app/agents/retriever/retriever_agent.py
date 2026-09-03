"""
Retriever Agent

Its ONLY job is: question in, evidence out. It never generates an
answer — that's the Analyst/Answer agents' job.

Pipeline: rewrite -> (semantic + keyword search in parallel) -> rerank
-> select context.
"""
import logging

from app.agents.retriever.tools.context_selector import select_context
from app.agents.retriever.tools.keyword_search import keyword_search
from app.agents.retriever.tools.query_rewriter import rewrite_query
from app.agents.retriever.tools.reranker import rerank
from app.ingestion.document import Document
from app.knowledge_store.vector_store import similarity_search

logger = logging.getLogger(__name__)

def retrieve_evidence(
    question: str,
    history: str = "",
    metadata_filter: dict | None = None,
    candidate_k: int = 10,
    final_k: int = 6,
) -> list[Document]:

    """
    Runs the full Retriever pipeline and returns the final, curated
    evidence bundle ready to hand to the Analyst Agent.
    """
    # Step 1: clean up the question
    query = rewrite_query(question, history=history)
    logger.info("query rewritten: %r -> %r", question, query)

    # Step 2: cast a wide net with two complementary search strategies
    semantic_hits = similarity_search(query, k=candidate_k, metadata_filter=metadata_filter)
    keyword_hits = keyword_search(query, k=candidate_k)

    # Step 3: merge candidate pools, de-duplicating by content
    seen = set()
    candidates: list[Document] = []
    for doc in semantic_hits + keyword_hits:
        fingerprint = doc.content[:150]
        if fingerprint not in seen:
            seen.add(fingerprint)
            candidates.append(doc)

    logger.info("retrieved %d unique candidates (%d semantic, %d keyword)",
                len(candidates), len(semantic_hits), len(keyword_hits))

    if not candidates:
        return []

    # Step 4: rerank the merged pool for actual relevance to the query
    reranked = rerank(query, candidates)

    # Step 5: trim down to a compact, diverse final evidence set
    final_evidence = select_context(reranked, max_chunks=final_k)

    logger.info("final evidence: %d chunks selected", len(final_evidence))
    return final_evidence
