# app/validation.py
from pathlib import Path

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".pptx", ".ppt", ".wav",
                       ".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rb", ".rs", ".cs"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB


def validate_upload(filename: str, size_bytes: int) -> None:
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {ext or '(none)'}")
    if size_bytes <= 0:
        raise ValueError("Uploaded file is empty")
    if size_bytes > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"File exceeds max size of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB")


def validate_url(url: str) -> None:
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("URL must start with http:// or https://")
