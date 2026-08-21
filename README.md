# Evaluator–Generator RAG Platform — Architecture

## Overview

This platform answers user questions using retrieval-augmented generation,
but instead of returning the first generated answer, it routes every answer
through a self-critique loop: a Generator LLM produces an answer, an
Evaluator LLM scores it against the retrieved context, and the Generator
revises based on feedback until the answer is accepted or a hard loop
limit is hit.

## Components

### Ingestion pipeline (`app/ingestion/`)

- `loaders.py` — one function per source type (PDF, DOCX, TXT, code, PPTX,
  web page, Wikipedia, WAV via Whisper transcription). Each loader returns
  a list of `Document` objects tagged with `source_type` and `source_name`
  metadata.
- `pipeline.py::ingest_source()` — dispatches to the right loader based on
  file extension or explicit `SourceType`, then chunks (`chuncking.py`,
  500 chars / 100 overlap by default) and stores the chunks via
  `add_documents()`. Raises `ValueError` for bad/empty input and
  `RuntimeError` for downstream failures, so the API layer can map these
  to 400 vs 502 responses.

### Knowledge store (`app/ingestion/vector_store.py`)

Chroma-backed vector store. Embeddings are produced by
`CachedEmbeddings` (see caching below), which wraps a HuggingFace
sentence-transformer model.

### Generator agent (`app/agents/generator/`)

- Retrieves top-k chunks for the question via similarity search.
- Builds a prompt strictly instructing it to answer only from retrieved
  context, and to say explicitly when the context is insufficient rather
  than fabricate an answer.
- If evaluator feedback is present, it's injected into the prompt so the
  Generator revises rather than regenerating from scratch.
- Backed by `GenerateMemory`, an isolated `InMemoryChatMessageHistory`
  instance created fresh per question/session.

### Evaluator agent (`app/agents/evaluator/`)

- Scores the Generator's answer on four axes (accuracy, relevance,
  completeness, grounding), each 0.0–1.0, via a `PydanticOutputParser`
  that forces structured output.
- `accepted = true` only if every score is ≥ 0.7.
- When rejected, returns specific feedback text that is fed back to the
  Generator on the next loop iteration.
- Backed by `EvaluatorMemory`, its own isolated
  `InMemoryChatMessageHistory` instance.

### Memory isolation

`GenerateMemory` and `EvaluatorMemory` are separate classes, each wrapping
its own `InMemoryChatMessageHistory`. They are instantiated once per
question inside the orchestrator (`workflow.py`) and never passed to each
other's agent functions — there is no code path by which the Generator's
prompt can see Evaluator memory contents or vice versa. Each memory only
stores what its own agent needs (Generator: questions + answers;
Evaluator: verdicts + feedback).

### Orchestrator (`app/orchestrator/workflow.py`)

Implements the loop as an LCEL `RunnableLambda` (`workflow_chain`) wrapping
a bounded `for` loop:

`max_feedback_loops` (default 4, from `Settings`) is enforced with a plain
loop counter. If loop 4 is reached without acceptance, the last generated
answer is returned along with `max_loops_reached=True` and a disclaimer
string, rather than raising an error — the system always returns
_something_ usable to the user.

Wrapping the loop in `RunnableLambda` keeps it composable with other LCEL
chains (`.invoke`, `.batch`, `.stream`, piping into further Runnables)
while keeping the actual iteration logic as an explicit, easily-debuggable
Python loop rather than forcing recursive `RunnableBranch` chains to
express bounded iteration.

### Caching (`app/cache/redis_cache.py`)

Redis caches four categories of data, each keyed by a SHA-256 hash of its
inputs so identical requests hit cache instead of recomputing:

| Cache               | Key inputs                    | Avoids                                                 |
| ------------------- | ----------------------------- | ------------------------------------------------------ |
| Embeddings          | raw text                      | Re-embedding identical chunks/queries                  |
| Retrieval results   | query + k                     | Re-running vector search for repeat questions          |
| Generator responses | question + context + feedback | Re-calling the LLM for an identical generation request |
| Evaluations         | question + answer + context   | Re-scoring an identical answer                         |

Because cache keys include the _context_ and _feedback_ (not just the raw
question), a changed retrieval result or a new feedback string naturally
produces a different cache key — stale answers are not returned when the
underlying state changes. All Redis calls are wrapped in try/except so a
Redis outage degrades to "always miss" rather than crashing the app.

### API layer (`app/main.py`)

FastAPI endpoints:

- `POST /ingest/file` — multipart upload, validated (extension + size) via
  `app/validation.py`, routed through the ingestion pipeline.
- `POST /ingest/url`, `POST /ingest/wikipedia` — same pipeline, non-file
  sources.
- `POST /ask` — runs the full Generator↔Evaluator loop and returns the
  final answer, acceptance status, loop count, disclaimer (if
  max-loops was hit), and the full evaluation trace for transparency.
- `GET /health` — liveness check.

Errors are mapped to HTTP status codes: `ValueError` (bad input) → 400,
`RuntimeError` (LLM/storage failure) → 502.

### UI (`app/ui/streamlit_app.py`)

Streamlit front end with an ingestion tab (file upload / URL / Wikipedia)
and a Q&A tab that displays the final answer plus a per-loop evaluation
trace (scores + feedback), so the self-correction process is visible, not
just the end result.

### Logging (`app/logging_config.py`)

Structured stdout logging configured once at startup. Each loop iteration
logs its scores and acceptance decision; the cache module logs hit/miss
per key; the ingestion pipeline logs source, chunk count, and failures.

## Data flow summary
