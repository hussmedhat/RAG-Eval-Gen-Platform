from pathlib import Path
from typing import Any, cast

import whisper

from app.ingestion.document import Document

_model_cache: dict[str, Any] = {}


def _get_model(model_name: str) -> Any:
    if model_name not in _model_cache:
        _model_cache[model_name] = whisper.load_model(model_name)
    return _model_cache[model_name]


def load_audio(file_path: str, model_name: str = "base") -> list[Document]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    model = _get_model(model_name)
    result: dict[str, Any] = model.transcribe(str(path))

    documents: list[Document] = []
    raw_segments = cast(list[dict[str, Any]], result.get("segments", []))

    for segment in raw_segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue

        start_time = float(segment.get("start", 0.0))
        end_time = float(segment.get("end", 0.0))

        documents.append(
            Document(
                content=text,
                metadata={
                    "source_type": "audio",
                    "source_name": path.name,
                    "timestamp_start": round(start_time, 2),
                    "timestamp_end": round(end_time, 2),
                },
            )
        )

    return documents
