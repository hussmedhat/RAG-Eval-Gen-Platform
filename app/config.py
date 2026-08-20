from functools import lru_cache
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
import os

load_dotenv()

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- LLM provider: OpenRouter ---
    openrouter_api_key: str= os.getenv("OPENROUTER_API_KEY")
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL")

    generator_model: str = os.getenv("GENERATOR_MODEL")
    evaluator_model: str = os.getenv("EVALUATOR_MODEL")

    # --- Embeddings ---
    embedding_model: str = os.getenv("EMBEDDING_MODEL")

    # --- Vector store ---
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH")
    vector_store_collection: str = os.getenv("VECTOR_STORE_COLLECTION")

    # --- Redis cache ---
    redis_url: str = os.getenv("REDIS_URL")
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS"))

    # --- Orchestration ---
    max_feedback_loops: int = int(os.getenv("MAX_FEEDBACK_LOOPS"))

    # --- Speech-to-text ---
    whisper_model: str = os.getenv("WHISPER_MODEL")



@lru_cache
def get_settings() -> Settings:
    return Settings()