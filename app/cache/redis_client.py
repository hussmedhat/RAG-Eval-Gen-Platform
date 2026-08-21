import hashlib
import json
import logging
from typing import cast
import redis

from app.config import get_settings

logger = logging.getLogger(__name__)
_redis_client: redis.Redis | None = None

def get_redis_client() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = redis.Redis.from_url(settings.redis_url,decode_responses=True)
    return _redis_client

def _hash_key(*parts : str) -> str | None:
    """Generate a SHA256 hash for the given parts to use as a cache key."""
    if not parts:
        return None
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()

def _get(key: str) -> str | None:
    try:
        raw_val = get_redis_client().get(key)
        val = cast(str | None, raw_val)
        logger.info("cache %s key=%s", "hit" if val else "miss", key)
        return val
    except redis.RedisError as e:
        logger.warning("redis get failed, bypassing cache: %s", e)
        return None

def _set(key : str, value : str, ttl_seconds : int) -> None:
    try:
        settings = get_settings()
        get_redis_client().set(key, value, ex=ttl_seconds or settings.cache_ttl_seconds)
    except redis.RedisError as e:
        logger.error(f"Redis error on SET for key {key}: {e}")


#  Embedding caching functions
def get_cached_embedding(text : str) -> list[float] | None:
    raw = _get(f"emb:{_hash_key(text)}")
    return json.loads(raw) if raw else None

def set_cached_embedding(text : str, embedding : list[float]) -> None:
    _set(f"emb:{_hash_key(text)}", json.dumps(embedding),ttl_seconds=get_settings().cache_ttl_seconds)

def get_cached_retrieval(query : str, k : int) -> list[dict] | None:
    raw = _get(f"ret:{_hash_key(query,str(k))}")
    return json.loads(raw) if raw else None

def set_cached_retrieval(query : str, k : int, documents : list[dict]) -> None:
    _set(f"ret:{_hash_key(query,str(k))}", json.dumps(documents), ttl_seconds=get_settings().cache_ttl_seconds)

# Generator LLM responses
def get_cached_generator_response(question : str, context : str,feedback : str | None) -> str | None:
    return _get(f"gen:{_hash_key(question,context,feedback or '')}")

def set_cached_generator_response(question : str, context : str, feedback : str | None, answer : str) -> None:
    _set(f"gen:{_hash_key(question,context,feedback or '')}", answer, ttl_seconds=get_settings().cache_ttl_seconds)

# Evaluation results

def get_cached_evaluation_result(question : str, answer : str, context : str) -> dict | None:
    raw = _get(f"eval:{_hash_key(question,answer,context)}")
    return json.loads(raw) if raw else None

def set_cached_evaluation_result(question : str, answer : str, context : str, result : dict) -> None:
    _set(f"eval:{_hash_key(question,answer,context)}", json.dumps(result), ttl_seconds=get_settings().cache_ttl_seconds)
