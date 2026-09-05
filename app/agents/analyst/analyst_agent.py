"""
Analyst Agent

Takes the evidence bundle from the Retriever and reasons over it:
runs the Table Extractor and Document Comparison tools automatically,
runs Data Analysis (via the Calculator) when the question needs a
numeric comparison, and judges whether the evidence is actually
sufficient to answer the question. If it isn't, it writes a targeted
follow-up query and sends it back to the Retriever via the Evidence
Request tool, then re-checks — bounded by max_feedback_loops so this
can never loop forever.

This agent never generates the final answer — that's the Answer
Agent's job. The Analyst only decides "do we have enough, and what
does it mean," and hands its findings downstream.
"""

import json
import logging
import re

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

from app.agents.analyst.tools.data_analysis import AnalysisResult, analyze
from app.agents.analyst.tools.document_comparison import ComparisonResult, compare_documents
from app.agents.analyst.tools.evidence_request import request_more_evidence
from app.agents.analyst.tools.table_extractor import ExtractedTable, extract_table
from app.config import get_settings
from app.ingestion.document import Document


logger = logging.getLogger(__name__)


# ---- Structured output for the sufficiency check ----

class AnalysisAssessment(BaseModel):
    sufficient: bool = Field(description="True if the evidence is enough to answer the question")
    missing_information: str = Field(
        default="", description="What's missing, empty string if sufficient"
    )
    follow_up_query: str = Field(
        default="", description="A specific, standalone query to send back to the "
        "Retriever for the missing information. Empty string if sufficient."
    )
    requires_numeric_comparison: bool = Field(
        description="True if answering requires comparing/ranking numeric values "
        "across labeled entities (e.g. models, sources, time periods)"
    )
    labeled_values: dict[str, float] = Field(
        default_factory=dict,
        description="Only fill this in if requires_numeric_comparison is true: "
        "a label -> numeric value mapping pulled from the evidence "
        "(e.g. {'Model A': 0.91, 'Model B': 0.87})",
    )
    key_facts: list[str] = Field(
        default_factory=list, description="Bullet-point facts from the evidence directly relevant to the question"
    )


_parser = PydanticOutputParser(pydantic_object=AnalysisAssessment)

_ASSESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are the Analyst in a RAG system. Given a question and retrieved "
     "evidence chunks, decide whether the evidence is sufficient to answer "
     "the question fully. Do not answer the question yourself — only assess "
     "the evidence. If something the question asks for is missing from the "
     "evidence, set sufficient=false and write a specific, standalone "
     "follow-up query that would retrieve the missing piece. If the question "
     "involves comparing or ranking numeric values across named entities, "
     "extract those values so they can be analyzed precisely.\n\n"
     "IMPORTANT: Output a single flat JSON object with the exact field names "
     "shown below, populated with real values. Do NOT wrap the object in a "
     "'properties' key, do NOT include 'type', 'required', or any other "
     "JSON-Schema keywords — those describe the shape, they are not part of "
     "your answer."),
    ("human", "Question: {question}\n\nEvidence:\n{evidence_block}\n\n{format_instructions}"),
]).partial(format_instructions=_parser.get_format_instructions())


def _build_assessment_llm() -> ChatOpenAI:
    """Returns just the raw LLM (no prompt, no parser attached) so _assess
    can pipe it with the prompt exactly once and inspect the raw text
    before parsing — needed for the repair step below."""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.evaluator_model,
        base_url=settings.openrouter_base_url,
        api_key=SecretStr(settings.openrouter_api_key),
        temperature=0,
        max_tokens=1024,
    )


def _repair_and_parse(raw_text: str) -> AnalysisAssessment:
    """Tries the normal parser first. If the model wrapped its answer in a
    JSON-Schema-shaped 'properties' key instead of a flat object (a known
    failure mode with weaker/free models echoing the format instructions
    back verbatim), unwrap it and retry once before giving up."""
    try:
        return _parser.parse(raw_text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {raw_text!r}")

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"Model output was not valid JSON: {raw_text!r}") from e

    if "properties" in parsed and isinstance(parsed["properties"], dict):
        parsed = parsed["properties"]

    return AnalysisAssessment.model_validate(parsed)


def _format_evidence(evidence: list[Document]) -> str:
    if not evidence:
        return "(no evidence retrieved)"
    return "\n\n".join(
        f"[{doc.metadata.get('source_name', 'unknown')}] {doc.content}"
        for doc in evidence
    )


def _assess(question: str, evidence: list[Document], max_retries: int = 2) -> AnalysisAssessment:
    llm = _build_assessment_llm()
    evidence_block = _format_evidence(evidence)

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 2):  # e.g. max_retries=2 -> 3 total attempts
        try:
            raw_response = (_ASSESS_PROMPT | llm).invoke({
                "question": question,
                "evidence_block": evidence_block,
            })
            content = raw_response.content
            raw_text = content if isinstance(content, str) else str(content)

            if not raw_text.strip():
                logger.warning(
                    "analyst assessment attempt %s returned an empty response, retrying", attempt
                )
                last_error = ValueError("Model returned an empty response")
                continue

            return _repair_and_parse(raw_text)

        except Exception as e:
            last_error = e
            logger.warning("analyst assessment attempt %s failed: %s", attempt, e)

    logger.exception("analyst assessment llm call failed after %s attempts", max_retries + 1)
    raise RuntimeError(f"Analyst assessment LLM call failed after retries: {last_error}") from last_error


# ---- Findings compiled once evidence is judged sufficient (or loops run out) ----

class AnalystFindings(BaseModel):
    key_facts: list[str]
    tables: list[ExtractedTable] = Field(default_factory=list)
    comparison: ComparisonResult | None = None
    data_analysis: dict | None = None  # AnalysisResult isn't a BaseModel, stored as dict


class AnalystOutput:
    def __init__(
        self,
        findings: AnalystFindings,
        evidence: list[Document],
        sufficient: bool,
        loops_used: int,
        max_loops_reached: bool,
    ):
        self.findings = findings
        self.evidence = evidence
        self.sufficient = sufficient
        self.loops_used = loops_used
        self.max_loops_reached = max_loops_reached


def _compile_findings(
    question: str,
    evidence: list[Document],
    assessment: AnalysisAssessment,
) -> AnalystFindings:
    tables = [t for t in (extract_table(doc.content) for doc in evidence) if t is not None]
    comparison = compare_documents(question, evidence)  # returns None if <2 sources, handled internally

    data_analysis_result: AnalysisResult | None = None
    if assessment.requires_numeric_comparison and assessment.labeled_values:
        data_analysis_result = analyze(assessment.labeled_values)

    return AnalystFindings(
        key_facts=assessment.key_facts,
        tables=tables,
        comparison=comparison,
        data_analysis=(
            {
                "stats": vars(data_analysis_result.stats),
                "ranking": [vars(p) for p in data_analysis_result.ranking],
            }
            if data_analysis_result is not None
            else None
        ),
    )


# ---- Main entrypoint: the sufficiency + feedback loop ----

def analyze_evidence(
    question: str,
    evidence: list[Document],
    history: str = "",
) -> AnalystOutput:
    """
    Runs the Analyst's sufficiency-check + feedback loop. Bounded by
    settings.max_feedback_loops, same cap the Generator/Evaluator loop
    used to enforce — reused here since it's the same "don't loop
    forever" concern.
    """
    if not question or not question.split():
        raise ValueError("Question must be a non-empty string.")

    settings = get_settings()
    max_loops = settings.max_feedback_loops or 4

    current_evidence = evidence
    assessment: AnalysisAssessment | None = None

    for loop in range(1, max_loops + 1):
        assessment = _assess(question, current_evidence)
        logger.info(
            "analyst loop %s: sufficient=%s missing=%r",
            loop, assessment.sufficient, assessment.missing_information,
        )

        if assessment.sufficient:
            findings = _compile_findings(question, current_evidence, assessment)
            return AnalystOutput(
                findings=findings,
                evidence=current_evidence,
                sufficient=True,
                loops_used=loop,
                max_loops_reached=False,
            )

        if not assessment.follow_up_query:
            # Model said insufficient but gave nothing to search for —
            # nothing more we can do, stop here rather than loop blindly.
            logger.warning("analyst marked evidence insufficient but gave no follow-up query")
            break

        evidence_count_before = len(current_evidence)
        current_evidence = request_more_evidence(
            assessment.follow_up_query, current_evidence, history=history,
        )

        if len(current_evidence) == evidence_count_before:
            logger.info(
                "analyst loop %s: follow-up query returned no new evidence, stopping early",
                loop,
            )
            break

    logger.warning("analyst reached max loops (%s) without sufficient evidence", max_loops)
    findings = _compile_findings(question, current_evidence, assessment) if assessment else AnalystFindings(key_facts=[])
    return AnalystOutput(
        findings=findings,
        evidence=current_evidence,
        sufficient=False,
        loops_used=max_loops,
        max_loops_reached=True,
    )
