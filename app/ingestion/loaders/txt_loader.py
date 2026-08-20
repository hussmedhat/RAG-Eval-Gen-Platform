#app/ingestion/loaders/txt_loader.py

from pathlib import Path

from app.ingestion.document import Document

def load_txt(file_path:str)-> list[Document]:
    path=Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    text= path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    return[
        Document(
            content=text,
            metadata={
                "source_type": "txt",
                "source_name": path.name,
            },
        )
    ]