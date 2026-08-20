from pathlib import Path

import whisper

from app.ingestion.document import Document

_model_cache: dict[str, "whisper.Whisper"] = {}


def _get_model(model_name: str):
    if model_name not in _model_cache:
        _model_cache[model_name] = whisper.load_model(model_name)
    return _model_cache[model_name]


def load_audio(file_path: str, model_name: str = "base") -> list[Document]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    model = _get_model(model_name)
    result = model.transcribe(str(path))

    documents: list[Document] = []
    for segment in result.get("segments", []):
        text = segment["text"].strip()
        if not text:
            continue

        documents.append(
            Document(
                content=text,
                metadata={
                    "source_type": "audio",
                    "source_name": path.name,
                    "timestamp_start": round(segment["start"], 2),
                    "timestamp_end": round(segment["end"], 2),
                },
            )
        )

    return documents