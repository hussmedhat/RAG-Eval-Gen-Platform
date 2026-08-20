#app/ingestion/loaders/pdf_loader.py

from pathlib import path
from pypdf import PdfReader

from app.ingestion.document import Document

def load_pdf(file_path: path) -> list[Document]:
    path = path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    reader= PdfReader(file_path)
    documents: list[Document] = []

    for page_number,page in enumerate(reader.pages):
        text= page.extract_text() or ""
        text= text.strip()
        if not text:
            continue
        documents.append(
            Document(
                content=text,
                metadata={
                    "source_type": "pdf",
                    "source_name": path.name,
                    "page": page_number + 1,
                    "total_pages": len(reader.pages)
                },
            )
        )

    return documents