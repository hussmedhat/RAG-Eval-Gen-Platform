import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.agents.generator.generator_memory import GenerateMemory
from app.config import get_settings
from app.cache.redis_client import set_cached_generator_response,get_cached_generator_response
from app.ingestion.document import Document
from app.knowledge_store.vector_store import similarity_search
from app.agents.prompts import GENERATOR_SYSTEM_PROMPT


logger = logging.getLogger(__name__)
_Prompt=ChatPromptTemplate.from_messages([
    ("system",GENERATOR_SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\n{feedback_section}"),
])


def _build_chain():
    settings=get_settings()
    llm=ChatOpenAI(
        model=settings.generator_model,
        base_url=settings.openrouter_base_url,
        api_key=SecretStr(settings.openrouter_api_key),
    )
    return _Prompt | llm | StrOutputParser()


def _format_context(retrieved: list[Document]) -> str:
    if not retrieved:
        return "(no relevant context retrieved)"
    return "\n\n".join(
        f"[{doc.metadata.get('source_name', 'unknown')}] {doc.content}"
        for doc in retrieved
    )


def generate_answer(
        question: str,
        feedback: str | None=None,
        memory: GenerateMemory | None=None,
        k: int=4,
)-> tuple[str,list[Document]]:
    if not question or not question.split():
        raise ValueError("Question must be a non-empty string.")

    retrieved=similarity_search(question,k=k)
    context=_format_context(retrieved)

    feedback_section=f"Evaluator feedback to section : {feedback}"if feedback else ""
    cached = get_cached_generator_response(question, context, feedback)
    if cached is not None:
        answer = cached
    else:
        chain=_build_chain()
        try:
            answer=chain.invoke({
                "context": context,
                "question":question,
                "feedback_section": feedback_section,
            })
        except Exception as e:
            logger.exception("generator llm call failed")
            raise RuntimeError(f"Generator LLM call failed: {e}") from e
        set_cached_generator_response(question,context,feedback,answer)
    if memory is not None:
        memory.add_user_message(question)
        memory.add_ai_message(answer)

    return answer,retrieved
