from functools import lru_cache
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
import os

load_dotenv()

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- LLM provider: OpenRouter ---
    openrouter_api_key: str= os.getenv("OPENROUTER_API_KEY") or ""
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL") or ""

    generator_model: str = os.getenv("GENERATOR_MODEL") or ""
    evaluator_model: str = os.getenv("EVALUATOR_MODEL") or ""

    # --- Embeddings ---
    embedding_model: str = os.getenv("EMBEDDING_MODEL") or ""

    # --- Vector store ---
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH") or ""
    vector_store_collection: str = os.getenv("VECTOR_STORE_COLLECTION") or ""

    # --- Redis cache ---
    redis_url: str = os.getenv("REDIS_URL") or ""
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    # --- Orchestration ---
    max_feedback_loops: int = int(os.getenv("MAX_FEEDBACK_LOOPS", "4"))  # Default to 4 if not set

    # --- Speech-to-text ---
    whisper_model: str = os.getenv("WHISPER_MODEL") or ""  # Default to "openai/whisper-large-v2" if not set

    # app/config.py — double check these three lines exist
    embedding_model: str = os.getenv("EMBEDDING_MODEL") or "liquid/lfm-2.5-embedGet-ChildItem -Force .envding-350m:free"
    embedding_base_url: str = os.getenv("EMBEDDING_BASE_URL") or ""

    transcription_model: str = os.getenv("TRANSCRIPTION_MODEL") or "openai/whisper-1"
    tesseract_cmd: str = os.getenv("TESSERACT_CMD") or ""





@lru_cache
def get_settings() -> Settings:
    return Settings()
