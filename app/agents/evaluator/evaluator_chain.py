import logging

from pydantic import BaseModel, Field, SecretStr
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.agents.evaluator.evaluator_memory import EvaluatorMemory
from app.agents.prompts import EVALUATOR_SYSTEM_PROMPT
from app.config import get_settings
from app.cache.redis_client import get_cached_evaluation_result, set_cached_evaluation_result
from app.ingestion.document import Document

logger = logging.getLogger(__name__)

class EvaluationResult(BaseModel):
    accepted: bool = Field(description="True only if all scores are >= 0.7")
    accuracy_score: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0)
    completeness_score: float = Field(ge=0.0, le=1.0)
    grounding_score: float = Field(ge=0.0, le=1.0)
    feedback: str = Field(
        description="Specific, actionable revision guidance. Empty string if accepted."
    )

_parser = PydanticOutputParser(pydantic_object=EvaluationResult)


_PROMPT = ChatPromptTemplate.from_messages([
    ("system", EVALUATOR_SYSTEM_PROMPT),
    (
        "human",
        "Context:\n{context}\n\nQuestion: {question}\n\n"
        "Generator's Answer: {answer}\n\n{format_instructions}",
    ),
]).partial(format_instructions=_parser.get_format_instructions())


def _build_chain():
    settings=get_settings()
    llm=ChatOpenAI(
        model=settings.evaluator_model,
        base_url=settings.openrouter_base_url,
        api_key=SecretStr(settings.openrouter_api_key),
    )
    return _PROMPT | llm | _parser


def _format_context(retrieved: list[Document]) -> str:
    if not retrieved:
        return "(no context was retrieved)"
    return "\n\n".join(
        f"[{doc.metadata.get('source_name', 'unknown')}] {doc.content}"
        for doc in retrieved
    )


def evaluate_answer(
        question:str,
        answer:str,
        retrieved:list[Document],
        memory:EvaluatorMemory|None=None,
)->EvaluationResult:
    context=_format_context(retrieved)
    cached = get_cached_evaluation_result(question, answer, context)
    if cached is not None:
        result = EvaluationResult(**cached)
    else:
        chain=_build_chain()
        try:
            result=chain.invoke({
                "context":context,
                "question":question,
                "answer":answer,
            })
        except Exception as e:
            logger.exception("evaluator llm call failed")
            raise RuntimeError(f"Evaluator LLM call failed: {e}") from e
        set_cached_evaluation_result(question, answer, context, result.dict())

    if memory is not None:
        memory.add_evaluation_input(f"Question: {question}\nAnswer graded: {answer}")
        memory.add_verdict(
            f"accepted={result.accepted}, feedback={result.feedback or '(none)'}"
        )
    return result
