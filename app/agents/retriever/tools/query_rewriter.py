"""
Tool 1 : Query Rewriter Tool

Takes the user's raw question (which may be vague, use pronouns, or be
a follow-up like "what about its downsides?") and rewrites it into a
clear, standalone, retrieval-friendly query using the LLM.
"""



from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.config import get_settings

_REWRITER_PROMPT = ChatPromptTemplate.from_messages([ ("system",
     "You rewrite user questions into clear, standalone search queries. "
     "Expand pronouns and vague references using the conversation history. "
     "Expand abbreviations. Output ONLY the rewritten query, nothing else."),
    ("human", "Conversation history:\n{history}\n\nOriginal question: {question}\n\nRewritten query:"),
])

def _build_chain():
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.generator_model,
        base_url=settings.openrouter_base_url,
        api_key = SecretStr(settings.openrouter_api_key),
        temperature=0,
    )
    return _REWRITER_PROMPT | llm | StrOutputParser()

def rewrite_query(question: str, history: str = "") -> str:
    """Returns a cleaned-up, standalone version of the question.
    Falls back to the original question if rewriting fails for any reason —
    a failed rewrite should never block retrieval entirely."""
    if not question or not question.split():
        raise ValueError("Question must be a non-empty string.")
    try:
        chain = _build_chain()
        rewritten_query = chain.invoke({"question": question, "history": history or "(none)"})
        return rewritten_query.strip() or question.strip()  # fallback to original question if rewriting fails
    except Exception:
        return question.strip()  # fallback to original question if rewriting fails
