"""
Agentic Workflow Orchestrator
"""

import logging

from langchain_core.runnables import RunnableLambda

from app.agents.answer.answer_agent import AnswerOutput, generate_final_answer
from app.agents.analyst.analyst_agent import analyze_evidence
from app.agents.retriever.retriever_agent import retrieve_evidence
from app.trace import get_trace, install_trace_handler, start_trace

logger = logging.getLogger(__name__)

install_trace_handler()  # attach once at import time


class AgenticWorkflowResult:
    def __init__(
        self,
        answer: str,
        sufficient: bool,
        loops_used: int,
        max_loops_reached: bool,
        num_citations: int,
        disclaimer: str | None,
        key_facts: list[str],
        tables: list[dict],
        comparison: dict | None,
        data_analysis: dict | None,
        trace: list[dict],
    ):
        self.answer = answer
        self.sufficient = sufficient
        self.loops_used = loops_used
        self.max_loops_reached = max_loops_reached
        self.num_citations = num_citations
        self.disclaimer = disclaimer
        self.key_facts = key_facts
        self.tables = tables
        self.comparison = comparison
        self.data_analysis = data_analysis
        self.trace = trace

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sufficient": self.sufficient,
            "loops_used": self.loops_used,
            "max_loops_reached": self.max_loops_reached,
            "num_citations": self.num_citations,
            "disclaimer": self.disclaimer,
            "key_facts": self.key_facts,
            "tables": self.tables,
            "comparison": self.comparison,
            "data_analysis": self.data_analysis,
            "trace": self.trace,
        }


def _run_agentic_workflow(question: str, history: str = "") -> AgenticWorkflowResult:
    start_trace()
    logger.info("starting agentic workflow for question: %r", question)

    evidence = retrieve_evidence(question, history=history)
    logger.info("retriever returned %d evidence chunks", len(evidence))

    analyst_output = analyze_evidence(question, evidence, history=history)
    logger.info(
        "analyst finished: sufficient=%s loops_used=%s max_loops_reached=%s",
        analyst_output.sufficient, analyst_output.loops_used, analyst_output.max_loops_reached,
    )

    answer_output: AnswerOutput = generate_final_answer(question, analyst_output)

    findings = analyst_output.findings
    trace = get_trace()
    return AgenticWorkflowResult(
        answer=answer_output.answer,
        sufficient=analyst_output.sufficient,
        loops_used=analyst_output.loops_used,
        max_loops_reached=analyst_output.max_loops_reached,
        num_citations=len(answer_output.cited_sources),
        disclaimer=answer_output.disclaimer,
        key_facts=findings.key_facts,
        tables=[t.model_dump() for t in findings.tables],
        comparison=findings.comparison.model_dump() if findings.comparison else None,
        data_analysis=findings.data_analysis,
        trace=trace,
    )


def _invoke_agentic_workflow(inputs: dict[str, str]) -> AgenticWorkflowResult:
    return _run_agentic_workflow(question=inputs["question"], history=inputs.get("history", ""))


agentic_workflow_chain = RunnableLambda(_invoke_agentic_workflow)


def run_agentic_workflow(question: str, history: str = "") -> AgenticWorkflowResult:
    if not question or not question.split():
        raise ValueError("Question must be a non-empty string.")
    return agentic_workflow_chain.invoke({"question": question, "history": history})
