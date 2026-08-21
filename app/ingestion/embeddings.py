# app/ingestion/embeddings.py
import logging

from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from app.cache.redis_client import get_cached_embedding, set_cached_embedding
from app.config import get_settings

logger = logging.getLogger(__name__)

_embedding_model: OpenAIEmbeddings | None = None


def _get_raw_model() -> OpenAIEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        settings = get_settings()
        _embedding_model = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=SecretStr(settings.openrouter_api_key),
            base_url=settings.embedding_base_url or None,
            tiktoken_enabled=False,      # <-- critical: skip tiktoken lookup for non-OpenAI model names
            check_embedding_ctx_length=False,  # <-- also relies on tiktoken; disable too
        )
    return _embedding_model


class CachedEmbeddings:
    def __init__(self, model: OpenAIEmbeddings):
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = [get_cached_embedding(t) for t in texts]
        missing_idx = [i for i, r in enumerate(results) if r is None]

        if missing_idx:
            missing_texts = [texts[i] for i in missing_idx]
            fresh = self._model.embed_documents(missing_texts)
            for i, emb in zip(missing_idx, fresh):
                results[i] = emb
                set_cached_embedding(texts[i], emb)

        return results  # type: ignore[return-value]

    def embed_query(self, text: str) -> list[float]:
        cached = get_cached_embedding(text)
        if cached is not None:
            return cached
        emb = self._model.embed_query(text)
        set_cached_embedding(text, emb)
        return emb


def get_embedding_model() -> CachedEmbeddings:
    return CachedEmbeddings(_get_raw_model())
