import requests
from bs4 import BeautifulSoup

from app.ingestion.document import Document

_STRIP_TAGS = ["nav", "footer", "header", "script", "style", "aside", "form"]


def load_web(url: str, timeout: int = 15) -> list[Document]:
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "rag-platform/1.0"})
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag_name in _STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else url
    text = soup.get_text(separator="\n")

    # collapse repeated blank lines left over after stripping tags
    lines = [line.strip() for line in text.splitlines()]
    clean_text = "\n".join(line for line in lines if line)

    if not clean_text:
        return []

    return [
        Document(
            content=clean_text,
            metadata={
                "source_type": "web",
                "source_name": url,
                "title": title,
            },
        )
    ]