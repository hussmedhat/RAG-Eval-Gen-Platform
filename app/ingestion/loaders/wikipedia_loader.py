# app/ingestion/loaders/wikipedia_loader.py
import requests

from app.ingestion.document import Document

WIKI_API_URL = "https://{lang}.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "RAG-Eval-Gen-Platform/1.0 (contact: your_email@example.com)"}


def _query_page(title: str, lang: str) -> dict | None:
    """Direct title lookup. Returns None if the page doesn't exist under this exact title."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|info",
        "explaintext": True,
        "inprop": "url",
        "titles": title,
        "redirects": 1,
    }
    resp = requests.get(WIKI_API_URL.format(lang=lang), params=params, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), None)
    if page is None or "missing" in page:
        return None
    return page


def _search_best_title(title: str, lang: str) -> str | None:
    """Full-text search fallback for when the exact title doesn't match
    (e.g. wrong capitalization: 'Amr diab' vs the real title 'Amr Diab')."""
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": title,
        "srlimit": 1,
    }
    resp = requests.get(WIKI_API_URL.format(lang=lang), params=params, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    results = resp.json().get("query", {}).get("search", [])
    if not results:
        return None
    return results[0]["title"]


def _fetch_page(title: str, lang: str) -> dict:
    page = _query_page(title, lang)
    if page is not None:
        return page

    # Exact/near-exact title lookup missed — fall back to search and retry.
    best_title = _search_best_title(title, lang)
    if best_title is None:
        raise FileNotFoundError(f"Wikipedia page not found: {title}")

    page = _query_page(best_title, lang)
    if page is None:
        raise FileNotFoundError(f"Wikipedia page not found: {title}")
    return page


def load_wikipedia(title: str, lang: str = "en") -> list[Document]:
    page = _fetch_page(title, lang)
    content = page.get("extract", "")
    page_title = page.get("title", title)
    page_url = page.get("fullurl", f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}")

    if not content.strip():
        raise FileNotFoundError(f"Wikipedia page has no content: {title}")

    documents: list[Document] = []
    section_name = "Introduction"
    buffer: list[str] = []
    idx = 0

    def flush():
        nonlocal idx
        text = "\n".join(buffer).strip()
        if text:
            idx += 1
            documents.append(
                Document(
                    content=text,
                    metadata={
                        "source_type": "wikipedia",
                        "source_name": page_title,
                        "url": page_url,
                        "section": section_name,
                        "section_index": idx,
                    },
                )
            )
        buffer.clear()

    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("==") and stripped.endswith("=="):
            flush()
            section_name = stripped.strip("= ").strip()
            continue
        buffer.append(stripped)
        if len(buffer) >= 3:
            flush()

    flush()
    return documents
