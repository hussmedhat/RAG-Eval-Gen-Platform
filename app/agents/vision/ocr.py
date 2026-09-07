"""
Image Upload: OCR

Extracts text from an uploaded image using Tesseract OCR via
pytesseract. The extracted text is treated as a normal ingestible
Document, so it flows through the same chunk -> embed -> store
pipeline as every other source type — an image is just another way
of getting text into the knowledge base.
"""
import logging

import pytesseract
from PIL import Image

from app.config import get_settings
from app.ingestion.document import Document

logger = logging.getLogger(__name__)

_configured = False


def _configure_tesseract() -> None:
    global _configured
    if _configured:
        return
    settings = get_settings()
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
    _configured = True


def extract_text_from_image(image_path: str, source_name: str = "uploaded_image") -> list[Document]:
    """Runs OCR on an image file and returns it as a single Document,
    ready for the normal chunk/embed/store pipeline. Raises ValueError
    if no text is detected — an image with no readable text isn't
    useful to ingest, so fail clearly rather than storing an empty
    chunk that would never be retrieved anyway."""
    _configure_tesseract()

    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image).strip()
    except Exception as e:
        logger.exception("OCR failed for %s", image_path)
        raise RuntimeError(f"OCR failed for '{image_path}': {e}") from e

    if not text:
        raise ValueError(f"No readable text found in image: {source_name}")

    logger.info("OCR extracted %d characters from %s", len(text), source_name)
    return [Document(
        content=text,
        metadata={"source_name": source_name, "source_type": "image_ocr"},
    )]
