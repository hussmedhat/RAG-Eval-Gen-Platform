import logging

from langchain_core.runnables import RunnableLambda

from app.agents.evaluator.evaluator_chain import evaluate_answer, EvaluationResult
from app.agents.evaluator.evaluator_memory import EvaluatorMemory
from app.agents.generator.generator_chain import generate_answer
from app.agents.generator.generator_memory import GenerateMemory
from app.config import get_settings

logger = logging.getLogger(__name__)

class WorkflowResult:
    def __init__(self, answer: str, accepted: bool , loops_used : int, max_loops_reached : bool, evaluation_history : list[EvaluationResult]):
        self.answer = answer
        self.accepted = accepted
        self.loops_used = loops_used
        self.max_loops_reached = max_loops_reached
        self.evaluation_history = evaluation_history

    def to_dict(self):
        return {
            "answer": self.answer,
            "accepted": self.accepted,
            "loops_used": self.loops_used,
            "max_loops_reached": self.max_loops_reached,
            "disclaimer": (
                "This answer could not be fully validated by the evaluator "
                "after the maximum number of iterations."
                if self.max_loops_reached else None
            ),
            "evaluation_history": [e.dict() for e in self.evaluation_history],
        }

def _run_loop(question : str) -> WorkflowResult:
    settings = get_settings()
    max_loops = settings.max_feedback_loops or 4

    gen_memory = GenerateMemory()
    eval_memory = EvaluatorMemory()

    feedback : str |None = None
    last_answer = ""
    evaluation_history : list[EvaluationResult] = []

    for loop in range(1, max_loops + 1):
        logger.info(f"Starting loop {loop} for question: {question}")
        answer , retrieved = generate_answer(question, feedback, gen_memory)
        last_answer = answer
        evaluation_result = evaluate_answer(question, answer, retrieved, eval_memory)
        evaluation_history.append(evaluation_result)
        logger.info(
            "loop %s scores acc=%.2f rel=%.2f comp=%.2f ground=%.2f accepted=%s",
            loop,
            evaluation_result.accuracy_score,
            evaluation_result.relevance_score,
            evaluation_result.completeness_score,
            evaluation_result.grounding_score,
            evaluation_result.accepted,
        )
        if evaluation_result.accepted:
            logger.info("Answer accepted after %s loops.", loop)
            return WorkflowResult(
                answer=answer,
                accepted=True,
                loops_used=loop,
                max_loops_reached=False,
                evaluation_history=evaluation_history,
            )
        feedback = evaluation_result.feedback

    logger.warning("max loops (%s) reached without acceptance", max_loops)
    return WorkflowResult(
        answer=last_answer,
        accepted=False,
        loops_used=max_loops,
        max_loops_reached=True,
        evaluation_history=evaluation_history,
    )

#LCEL wrapper for the whole workflow as it is itself a runnable that takes a question and returns a WorkflowResult
workflow_chain = RunnableLambda(_run_loop)

def run_workflow(question : str) -> WorkflowResult:
    if not question or not question.split():
        raise ValueError("Question must be a non-empty string.")
    return workflow_chain.invoke(question)
