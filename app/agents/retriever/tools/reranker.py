"""
Tool 5: Reranker

Semantic + keyword search together can return a wide, noisy candidate
pool (e.g. combined top 15-20). The reranker asks the LLM to re-score
each candidate's actual relevance to the specific question, then
reorders by that score. This is a second, more careful pass — slower
per-item than embedding similarity, so it only runs over the smaller
candidate pool, never the whole corpus.
"""

import logging
import time
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from app.config import get_settings
from app.ingestion.document import Document

logger = logging.getLogger(__name__)

_RERANK_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Rate how relevant the passage is to the question, from 0 to 10. "
     "Output ONLY the number, nothing else."),
    ("human", "Question: {question}\n\nPassage: {passage}\n\nRelevance score (0-10):"),
])


def _build_chain():
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.evaluator_model,
        base_url=settings.openrouter_base_url,
        api_key=SecretStr(settings.openrouter_api_key),
        temperature=0,
        max_tokens=500,
        extra_body={"reasoning": {"enabled": False}},
    )
    return _RERANK_PROMPT | llm | StrOutputParser()


def _extract_score(raw: str) -> float | None:
    """Parse a 0-10 score from the model's response. Returns None if
    empty or unparseable, rather than raising or silently zeroing."""
    raw = raw.strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _invoke_with_retry(chain, inputs: dict, source_name: str, max_attempts: int = 3) -> float | None:
    """Calls the rerank chain with retries + backoff. Returns a float
    score on success, or None if every attempt failed (exception, empty
    response, or unparseable text)."""
    for attempt in range(1, max_attempts + 1):
        try:
            raw = chain.invoke(inputs)
        except Exception:
            logger.exception("rerank: LLM call failed for candidate (source=%s, attempt %d/%d)",
                              source_name, attempt, max_attempts)
        else:
            score = _extract_score(raw)
            if score is not None:
                return score
            logger.warning("rerank: no parseable score for candidate (source=%s, attempt %d/%d): %r",
                            source_name, attempt, max_attempts, raw[:120])

        if attempt < max_attempts:
            time.sleep(1.5 * attempt)  # 1.5s, 3s between retries

    return None


def rerank(question: str, candidates: list[Document]) -> list[Document]:
    """Re-scores and reorders candidates by LLM-judged relevance.

    A candidate that has no score after all retries is marked score=None
    ("unscored") rather than 0.0 ("confirmed irrelevant") — select_context()
    skips None candidates instead of treating them as a genuine low-relevance
    verdict.
    """
    if not candidates:
        return []

    chain = _build_chain()
    scored: list[tuple[float, Document]] = []
    failed_count = 0

    for i, doc in enumerate(candidates):
        if i > 0:
            time.sleep(0.5)  # spread requests out to avoid bursting free-tier rate limits

        source_name = doc.metadata.get("source_name", "unknown")
        score = _invoke_with_retry(
            chain,
            {"question": question, "passage": doc.content[:800]},
            source_name,
        )

        if score is None:
            failed_count += 1

        doc.metadata = {**doc.metadata, "rerank_score": score}
        sort_key = score if score is not None else -1.0
        scored.append((sort_key, doc))

    if failed_count == len(candidates):
        logger.error(
            "rerank: ALL %d candidates failed to score after retries — "
            "check evaluator_model config / API health / rate limits",
            len(candidates),
        )
    elif failed_count:
        logger.warning("rerank: %d/%d candidates failed to score after retries",
                        failed_count, len(candidates))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored]
