# Evaluator–Generator RAG Platform — Contribution Summary

Cellula Technologies NLP Internship — Project 3

| Contributor | Scope of work |
|---|---|
| **Hussein Medhat** | Core agent logic: Generator, Evaluator, prompts, chunking, ingestion document model, embeddings, vector store, configuration, and the source loaders — the original architectural skeleton. |
| **Aly Adel** | Orchestration loop, Redis caching layer, ingestion pipeline dispatcher, FastAPI service layer, Streamlit UI. |

---

## 1. Hussein Medhat — Core Agent Logic

### `app/agents/generator/generator.py`
LCEL chain wiring context + question + feedback into the Generator LLM via `ChatOpenAI`/OpenRouter, with a `StrOutputParser` for plain-text answers.

### `app/agents/generator/generator_memory.py`
`GenerateMemory` class wrapping an isolated chat history for the Generator agent.

### `app/agents/evaluator/evaluator.py`
LCEL chain scoring a Generator answer against retrieved context using a `PydanticOutputParser` (`EvaluationResult`: accuracy, relevance, completeness, grounding, accepted, feedback).

### `app/agents/evaluator/evaluator_memory.py`
`EvaluatorMemory` class wrapping an isolated chat history for the Evaluator agent, kept fully separate from `GenerateMemory`.

### `app/agents/prompts.py`
`GENERATOR_SYSTEM_PROMPT` and `EVALUATOR_SYSTEM_PROMPT` — the grounding rule (answer only from context, say so explicitly if insufficient) and the 0.7-threshold scoring rubric.

### `app/ingestion/document.py`
`Document` dataclass (content + metadata) used across the whole ingestion pipeline.

### `app/ingestion/chuncking.py`
`chunk_documents()` using LangChain's `RecursiveCharacterTextSplitter`, tagging each chunk with `chunk_index`/`total_chunks` metadata.

### `app/ingestion/embeddings.py` (original)
First version of `get_embedding_model()`, returning a local `HuggingFaceEmbeddings` model.

### `app/ingestion/vector_store.py` (original)
`get_vector_store()`, `add_documents()`, and `similarity_search()` built on `langchain_chroma`.

### `app/config.py` (original)
`Settings` class (pydantic-settings) defining every environment-driven config value used by the rest of the app: OpenRouter, embeddings, vector store, Redis, orchestration, and Whisper.

### `app/ingestion/loaders/` (pdf, docx, txt, code, pptx, web, wikipedia)
One loader per supported input type, each returning `Document` objects tagged with source metadata for downstream chunking and retrieval.

---

## 2. Aly Adel — Orchestration, Caching, Ingestion Pipeline, API & UI

### `app/orchestrator/workflow.py`
The Evaluator–Generator feedback loop itself: instantiates isolated memory per question, loops Generator → Evaluator → accept/retry, enforces the 4-iteration cap from `settings.max_feedback_loops`, and returns a disclaimer on max-loop exit. Wrapped as an LCEL `RunnableLambda` so it composes with the rest of the LangChain pipeline.

### `app/cache/redis_client.py`
Full Redis caching layer: hashed cache keys for embeddings, retrieval results, Generator responses, and Evaluator results, with graceful fallback (cache miss, not a crash) if Redis is unreachable.

### `app/ingestion/pipeline.py`
`ingest_source()` dispatcher: routes an uploaded file, URL, or Wikipedia query to the correct loader by `SourceType`, then chunks and stores the result, with validation and error handling.

### `app/validation.py`
Upload validation: file extension allow-list, empty-file and max-size checks, URL scheme checks.

### `app/logging_config.py`
Centralized structured logging configuration used across ingestion, caching, and orchestration.

### `app/main.py`
The FastAPI service layer: `/ingest/file`, `/ingest/url`, `/ingest/wikipedia`, `/ask`, and `/health` endpoints, mapping `ValueError`/`RuntimeError` to proper 400/502 responses.

### `app/ui/streamlit_app.py`
The Streamlit front end: file/URL/Wikipedia ingestion tabs and a question-asking tab that displays the final answer, acceptance status, and the full per-loop evaluation trace.

---

## 3. Verified Working End-to-End

- PDF ingestion, embedded via the free OpenRouter model and stored in Chroma.
- A grounded, correct answer accepted by the Evaluator on the first loop (scores of 1.0 across all four criteria).
- A correct refusal when the retrieved context did not contain the requested fact, with the Generator explicitly stating the information was unavailable rather than fabricating an answer.
- The full request/response cycle through both the FastAPI `/docs` interface and the Streamlit UI.
