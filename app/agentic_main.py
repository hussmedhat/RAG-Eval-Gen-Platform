# app/agentic_main.py
import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.ingestion.pipeline import SourceType, infer_source_type, ingest_source
from app.logging_config import configure_logging
from app.orchestrator.agentic_workflow import run_agentic_workflow
from app.validation import validate_upload, validate_url
from app.agents.voice.transcriptions import transcribe_audio

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic RAG Platform (Retriever -> Analyst -> Answer)")


class URLIngestRequest(BaseModel):
    url: str


class WikipediaIngestRequest(BaseModel):
    query: str
    lang: str = "en"


class AskRequest(BaseModel):
    question: str
    history: str = ""

class TranscribeResponse(BaseModel):
    question: str


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a valid filename.")

    filename: str = file.filename
    contents = await file.read()

    try:
        validate_upload(filename, len(contents))
        source_type = infer_source_type(filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        n_chunks = ingest_source(tmp_path, source_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {"filename": filename, "chunks_ingested": n_chunks}


@app.post("/ingest/url")
async def ingest_url(req: URLIngestRequest):
    try:
        validate_url(req.url)
        n_chunks = ingest_source(req.url, SourceType.WEB)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {"url": req.url, "chunks_ingested": n_chunks}


@app.post("/ingest/wikipedia")
async def ingest_wikipedia(req: WikipediaIngestRequest):
    try:
        n_chunks = ingest_source(req.query, SourceType.WIKIPEDIA)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {"query": req.query, "chunks_ingested": n_chunks}


@app.post("/ask")
async def ask(req: AskRequest):
    try:
        result = run_agentic_workflow(req.question, history=req.history)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.exception("agentic workflow failed")
        raise HTTPException(status_code=502, detail=str(e))

    return result.to_dict()

@app.post("/voice/transcribe", response_model=TranscribeResponse)
async def voice_transcribe(file: UploadFile = File(...)):
    """Accepts a recorded audio clip and returns the transcribed text.
    The frontend then sends that text to /ask as a normal question —
    this endpoint does transcription only, it does not run the agentic
    pipeline itself."""
    contents = await file.read()
    audio_format = (file.filename or "audio.wav").rsplit(".", 1)[-1].lower()

    try:
        text = transcribe_audio(contents, audio_format=audio_format)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return TranscribeResponse(question=text)


@app.get("/health")
async def health():
    return {"status": "ok"}
