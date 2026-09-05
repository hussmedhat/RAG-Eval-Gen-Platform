"""
Agentic Workflow Orchestrator

Wires the three agents from the spec into a single flow:

    Retriever Agent -> Analyst Agent -> Answer Agent

Retriever finds evidence, Analyst judges sufficiency (looping back to
the Retriever for more evidence when needed) and extracts structured
findings, Answer Agent turns those findings into the final, cited
response. This is a separate orchestrator from workflow.py's
Generator/Evaluator self-critique loop — the two are independent
pipelines, not composed together.
"""

import logging

from langchain_core.runnables import RunnableLambda

from app.agents.answer.answer_agent import AnswerOutput, generate_final_answer
from app.agents.analyst.analyst_agent import analyze_evidence
from app.agents.retriever.retriever_agent import retrieve_evidence

logger = logging.getLogger(__name__)


class AgenticWorkflowResult:
    def __init__(
        self,
        answer: str,
        sufficient: bool,
        loops_used: int,
        max_loops_reached: bool,
        num_citations: int,
        disclaimer: str | None,
    ):
        self.answer = answer
        self.sufficient = sufficient
        self.loops_used = loops_used
        self.max_loops_reached = max_loops_reached
        self.num_citations = num_citations
        self.disclaimer = disclaimer

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sufficient": self.sufficient,
            "loops_used": self.loops_used,
            "max_loops_reached": self.max_loops_reached,
            "num_citations": self.num_citations,
            "disclaimer": self.disclaimer,
        }


def _run_agentic_workflow(question: str, history: str = "") -> AgenticWorkflowResult:
    logger.info("starting agentic workflow for question: %r", question)

    evidence = retrieve_evidence(question, history=history)
    logger.info("retriever returned %d evidence chunks", len(evidence))

    analyst_output = analyze_evidence(question, evidence, history=history)
    logger.info(
        "analyst finished: sufficient=%s loops_used=%s max_loops_reached=%s",
        analyst_output.sufficient, analyst_output.loops_used, analyst_output.max_loops_reached,
    )

    answer_output: AnswerOutput = generate_final_answer(question, analyst_output)

    return AgenticWorkflowResult(
        answer=answer_output.answer,
        sufficient=analyst_output.sufficient,
        loops_used=analyst_output.loops_used,
        max_loops_reached=analyst_output.max_loops_reached,
        num_citations=len(answer_output.cited_sources),
        disclaimer=answer_output.disclaimer,
    )


# LCEL wrapper, same pattern as workflow.py's workflow_chain
agentic_workflow_chain = RunnableLambda(lambda inputs: _run_agentic_workflow(**inputs))


def run_agentic_workflow(question: str, history: str = "") -> AgenticWorkflowResult:
    if not question or not question.split():
        raise ValueError("Question must be a non-empty string.")
    return agentic_workflow_chain.invoke({"question": question, "history": history})