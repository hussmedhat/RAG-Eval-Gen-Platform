import re
from pathlib import Path

from app.ingestion.document import Document

# Lightweight, regex-based block starts per language — not a full parser,
# but good enough to avoid slicing a function in half. Swap for a proper
# tree-sitter splitter later if precision becomes an issue.
_BLOCK_PATTERNS = {
    ".py": re.compile(r"^(def |class |async def )"),
    ".js": re.compile(r"^(function |class |const \w+\s*=\s*\(|export )"),
    ".ts": re.compile(r"^(function |class |const \w+\s*=\s*\(|export )"),
    ".java": re.compile(r"^\s*(public|private|protected).*\b(class|\w+\s*\()"),
    ".go": re.compile(r"^func "),
}


def load_code(file_path: str) -> list[Document]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Code file not found: {file_path}")

    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return []

    pattern = _BLOCK_PATTERNS.get(path.suffix)

    if not pattern:
        return [
            Document(
                content=text.strip(),
                metadata={
                    "source_type": "code",
                    "source_name": path.name,
                    "language": path.suffix.lstrip("."),
                },
            )
        ]

    lines = text.splitlines()
    blocks: list[list[str]] = []
    current_block: list[str] = []

    for line in lines:
        if pattern.match(line) and current_block:
            blocks.append(current_block)
            current_block = [line]
        else:
            current_block.append(line)

    if current_block:
        blocks.append(current_block)

    documents: list[Document] = []
    for idx, block_lines in enumerate(blocks, start=1):
        block_text = "\n".join(block_lines).strip()
        if not block_text:
            continue

        # best-effort function/class name for metadata
        name_match = pattern.search("\n".join(block_lines))
        block_name = block_lines[0].strip()[:60] if block_lines else f"block_{idx}"

        documents.append(
            Document(
                content=block_text,
                metadata={
                    "source_type": "code",
                    "source_name": path.name,
                    "language": path.suffix.lstrip("."),
                    "block_index": idx,
                    "block_name": block_name,
                },
            )
        )

    return documents