from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.agents.generator.generator_memory import GenerateMemory
from app.config import get_settings
from app.ingestion.document import Document
from app.knowledge_store.vector_store import similarity_search
from app.agents.prompts import GENERATOR_SYSTEM_PROMPT


_Prompt=ChatPromptTemplate.from_messages([
    ("system",GENERATOR_SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\n{feedback_section}"),
])


def _build_chain():
    settings=get_settings()
    llm=ChatOpenAI(
        model_name=settings.generator_model,
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
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
    retrieved=similarity_search(question,k=k)
    context=_format_context(retrieved)

    feedback_section=f"Evaluator feedback to section : {feedback}"if feedback else ""

    chain=_build_chain()
    answer=chain.invoke({
        "context": context,
        "question":question,
        "feedback_section": feedback_section,
    })

    if memory is not None:
        memory.add_user_message(question)
        memory.add_ai_message(answer)

    return answer,retrieved