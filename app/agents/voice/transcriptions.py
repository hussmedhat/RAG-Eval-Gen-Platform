"""
Voice Query: Transcription

Sends recorded audio bytes to OpenRouter's dedicated speech-to-text
endpoint (POST /api/v1/audio/transcriptions) and returns the
transcribed text, which then flows into the normal /ask pipeline as
if the user had typed it.
"""
import base64
import logging

import requests

from app.config import get_settings

logger = logging.getLogger(__name__)


def transcribe_audio(audio_bytes: bytes, audio_format: str = "wav") -> str:
    """Transcribes recorded audio into text via OpenRouter's Whisper
    endpoint. Raises RuntimeError on any failure (network, auth, bad
    audio) so the caller can surface a clear error instead of silently
    returning an empty question."""
    settings = get_settings()

    if not audio_bytes:
        raise ValueError("No audio data received")

    encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            json={
                "model": settings.transcription_model,
                "input_audio": {"data": encoded_audio, "format": audio_format},
            },
            timeout=60,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        logger.exception("transcription request failed")
        raise RuntimeError(f"Transcription request failed: {e}") from e

    data = response.json()
    text = data.get("text", "").strip()

    if not text:
        raise ValueError("Transcription returned no text — try recording again")

    logger.info("transcribed audio to: %r", text)
    return text
