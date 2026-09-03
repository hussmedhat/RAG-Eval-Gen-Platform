"""
Tool 5: Reranker

Semantic + keyword search together can return a wide, noisy candidate
pool (e.g. combined top 15-20). The reranker asks the LLM to re-score
each candidate's actual relevance to the specific question, then
reorders by that score. This is a second, more careful pass — slower
per-item than embedding similarity, so it only runs over the smaller
candidate pool, never the whole corpus.
"""


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from app.config import get_settings
from app.ingestion.document import Document

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
    )
    return _RERANK_PROMPT | llm | StrOutputParser()

def rerank(question : str,candidates : list[Document]) -> list[Document]:
    """Re-scores and reorders candidates by LLM-judged relevance.
    If a candidate fails to score (bad LLM output), it's kept but pushed
    to the bottom rather than dropped — a formatting hiccup shouldn't
    silently delete evidence."""
    if not candidates:
        return []

    chain = _build_chain()
    scored : list[tuple[float,Document]] = []

    for doc in candidates:
        try:
            raw_score = chain.invoke({"question": question, "passage": doc.content[:800]})
            score = float(raw_score)
        except (ValueError,Exception):
            score = 0.0  # push to bottom if scoring fails
        doc.metadata = {**doc.metadata, "rerank_score": score}
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored]


