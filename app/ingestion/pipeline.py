import logging
from enum import Enum

from app.config import get_settings
from app.ingestion.chuncking import chunk_documents
from app.ingestion.document import Document
from app.ingestion.loaders.audio_loader import load_audio
from app.ingestion.loaders.code_loader import load_code
from app.ingestion.loaders.docx_loader import load_docx
from app.ingestion.loaders.pptx_loader import load_pptx
from app.ingestion.loaders.txt_loader import load_txt
from app.ingestion.loaders.pdf_loader import load_pdf
from app.ingestion.loaders.audio_loader import load_audio
from app.ingestion.loaders.web_loader import load_web
from app.ingestion.loaders.wikipedia_loader import load_wikipedia

from app.knowledge_store.vector_store import add_documents

logger = logging.getLogger(__name__)

CODE_EXTENSIONS = {".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rb", ".rs", ".cs"}

class SourceType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    CODE = "code"
    PPTX = "pptx"
    WEB = "web"
    WIKIPEDIA = "wikipedia"
    WAV = "wav"


def _load(source: str, source_type: SourceType) -> list[Document]:
    settings = get_settings()

    loaders = {
        SourceType.PDF: lambda: load_pdf(source),
        SourceType.DOCX: lambda: load_docx(source),
        SourceType.TXT: lambda: load_txt(source),
        SourceType.CODE: lambda: load_code(source),
        SourceType.PPTX: lambda: load_pptx(source),
        SourceType.WEB: lambda: load_web(source),
        SourceType.WIKIPEDIA: lambda: load_wikipedia(source),
        SourceType.WAV: lambda: load_audio(source, settings.whisper_model),
    }

    loader = loaders.get(source_type)
    if loader is None:
        raise ValueError(f"Unsupported source type: {source_type}")
    return loader()

def ingest_source(source: str, source_type: SourceType, chunk_size: int = 500, chunk_overlap: int = 100) -> int:
    """
    Loads a source, chunks it, embeds+stores it.
    Returns the number of chunks stored.
    Raises ValueError/RuntimeError on failure — caller (API layer) should
    catch and translate into an HTTP error response.
    """
    logging.info(f"Starting ingestion for source: {source}, type: {source_type}")
    try:
        documents = _load(source, source_type)
    except ValueError:
        raise
    except Exception as e:
        logger.exception(f"Failed to load source: {source}, type: {source_type}")
        raise RuntimeError(f"Failed to load source: {e}") from e
    if not documents:
        raise ValueError(f"No content loaded from source: {source}, type: {source_type}")
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError(f"Chunking produced no chunks for source: {source}, type: {source_type}")
    try:
        add_documents(chunks)
    except Exception as e:
        logger.exception("failed to store chunks for source %r", source)
        raise RuntimeError(f"Failed to store embeddings for '{source}': {e}") from e
    logger.info("ingested %d chunks from %r", len(chunks), source)
    return len(chunks)

def infer_source_type(filename: str) -> SourceType:
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    mapping = {
        ".pdf": SourceType.PDF,
        ".docx": SourceType.DOCX,
        ".txt": SourceType.TXT,
        ".pptx": SourceType.PPTX,
        ".ppt": SourceType.PPTX,
        ".wav": SourceType.WAV,
    }
    if ext in mapping:
        return mapping[ext]
    if ext in CODE_EXTENSIONS:
        return SourceType.CODE
    raise ValueError(f"Cannot infer source type for file: {filename}")
