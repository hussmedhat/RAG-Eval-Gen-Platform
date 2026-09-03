"""
Analyst Tool 3: Document Comparison

Compares information across evidence chunks that come from different
source documents. Groups the evidence bundle by source_name (set by
the Retriever/ingestion pipeline in each Document's metadata), then
asks the LLM to identify agreements, contradictions, and unique points
per source

"""

from collections import defaultdict

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

from app.config import get_settings
from app.ingestion.document import Document


class ComparisonResult(BaseModel):
    agreements: list[str] = Field(description="Facts/claims that multiple sources agree on")
    contradictions: list[str] = Field(description="Facts/claims where sources disagree, empty if none")
    source_specific_points: dict[str, list[str]] = Field(
        description="Points that appear in only one source, keyed by source name"
    )


_parser = PydanticOutputParser(pydantic_object=ComparisonResult)

_COMPARE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You compare information across multiple document sources for the "
     "given question. Identify what the sources agree on, where they "
     "contradict each other, and any points that appear in only one source."),
    ("human",
     "Question: {question}\n\nSources:\n{sources_block}\n\n{format_instructions}"),
]).partial(format_instructions=_parser.get_format_instructions())


def _group_by_source(evidence: list[Document]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for doc in evidence:
        source = doc.metadata.get("source_name", "unknown")
        grouped[source].append(doc.content)
    return grouped

def _build_chain():
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.evaluator_model,
        base_url=settings.openrouter_base_url,
        api_key=SecretStr(settings.openrouter_api_key),
        temperature=0,
    )
    return _COMPARE_PROMPT | llm | _parser


def compare_documents(question: str, evidence: list[Document]) -> ComparisonResult | None:
    """Compares evidence across distinct sources"""
    grouped = _group_by_source(evidence)

    if len(grouped) < 2:
        return None  # nothing to compare — only one source present

    sources_block = "\n\n".join(
        f"[{source}]\n" + "\n".join(chunks)
        for source, chunks in grouped.items()
    )

    chain = _build_chain()
    return chain.invoke({"question": question, "sources_block": sources_block})
