"""
Answer Agent

Takes the Analyst's compiled findings plus the evidence bundle they
were built from and writes the final, user-facing answer: grounded
in the evidence, citing exact sources by document + page, and
weaving in any tables or rankings the Analyst extracted when the
question calls for a comparison.

This agent never re-judges evidence sufficiency — that's already been
decided by the Analyst (AnalystOutput.sufficient / max_loops_reached).
Its only job is presentation: turn findings into prose, cite it, and
attach sources.
"""

import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.agents.analyst.analyst_agent import AnalystOutput
from app.agents.answer.tools.citation_formatter import format_citations
from app.agents.answer.tools.response_formatter import assemble_response, build_reference_context
from app.agents.answer.tools.source_formatter import format_sources
from app.agents.prompts import ANSWER_SYSTEM_PROMPT
from app.config import get_settings
from app.ingestion.document import Document
from app.agents.prompts import ANSWER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ANSWER_SYSTEM_PROMPT),
    (
        "human",
        "Question: {question}\n\n"
        "Key facts noted by the Analyst:\n{key_facts}\n\n"
        "Evidence chunks:\n{evidence_block}\n\n"
        "Structured data available (tables / rankings):\n{reference_context}\n\n"
        "{insufficiency_note}",
    ),
])


def _build_chain():
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.generator_model,
        base_url=settings.openrouter_base_url,
        api_key=SecretStr(settings.openrouter_api_key),
    )
    return _PROMPT | llm | StrOutputParser()


def _format_evidence(evidence: list[Document]) -> str:
    if not evidence:
        return "(no evidence retrieved)"
    lines = []
    for i, doc in enumerate(evidence, start=1):
        source = doc.metadata.get("source_name", "unknown")
        locator = doc.metadata.get("page") or doc.metadata.get("slide_number") or doc.metadata.get("section")
        header = f"[E{i}] {source}" + (f", p.{locator}" if locator else "")
        lines.append(f"{header}\n{doc.content}")
    return "\n\n".join(lines)


def _format_key_facts(key_facts: list[str]) -> str:
    if not key_facts:
        return "(none extracted)"
    return "\n".join(f"- {fact}" for fact in key_facts)


class AnswerOutput:
    def __init__(self, answer: str, cited_sources: list[Document], disclaimer: str | None):
        self.answer = answer
        self.cited_sources = cited_sources
        self.disclaimer = disclaimer

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "num_citations": len(self.cited_sources),
            "disclaimer": self.disclaimer,
        }


def generate_final_answer(question: str, analyst_output: AnalystOutput) -> AnswerOutput:
    """Writes the final, cited answer from the Analyst's output."""
    if not question or not question.split():
        raise ValueError("Question must be a non-empty string.")

    evidence = analyst_output.evidence
    findings = analyst_output.findings

    evidence_block = _format_evidence(evidence)
    reference_context = build_reference_context(findings.tables, findings.data_analysis)
    key_facts = _format_key_facts(findings.key_facts)

    insufficiency_note = (
        "Note: the Analyst could not confirm the evidence gathered is fully sufficient "
        "for this question after the maximum number of retrieval attempts. Answer with "
        "what's available and be explicit about any gaps."
        if analyst_output.max_loops_reached else ""
    )

    chain = _build_chain()
    try:
        raw_answer = chain.invoke({
            "question": question,
            "key_facts": key_facts,
            "evidence_block": evidence_block,
            "reference_context": reference_context,
            "insufficiency_note": insufficiency_note,
        })
    except Exception as e:
        logger.exception("answer agent llm call failed")
        raise RuntimeError(f"Answer Agent LLM call failed: {e}") from e

    formatted_answer, cited_evidence = format_citations(raw_answer, evidence)
    sources_block = format_sources(cited_evidence)
    final_answer = assemble_response(formatted_answer, sources_block)

    disclaimer = (
        "This answer could not be fully validated with sufficient evidence after the "
        "maximum number of retrieval attempts."
        if analyst_output.max_loops_reached else None
    )

    return AnswerOutput(answer=final_answer, cited_sources=cited_evidence, disclaimer=disclaimer)