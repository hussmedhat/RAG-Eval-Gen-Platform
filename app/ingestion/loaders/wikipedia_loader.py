# app/ingestion/loaders/wikipedia_loader.py
import wikipedia

from app.ingestion.document import Document


def load_wikipedia(title: str, lang: str = "en") -> list[Document]:
    wikipedia.set_lang(lang)

    try:
        page = wikipedia.page(title, auto_suggest=False)
    except wikipedia.exceptions.DisambiguationError as e:
        page = wikipedia.page(e.options[0], auto_suggest=False)
    except wikipedia.exceptions.PageError:
        results = wikipedia.search(title)
        if not results:
            raise FileNotFoundError(f"Wikipedia page not found: {title}")
        try:
            page = wikipedia.page(results[0], auto_suggest=False)
        except wikipedia.exceptions.DisambiguationError as e:
            page = wikipedia.page(e.options[0], auto_suggest=False)
        except wikipedia.exceptions.PageError:
            raise FileNotFoundError(f"Wikipedia page not found: {title}")

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
                        "source_name": page.title,
                        "url": page.url,
                        "section": section_name,
                        "section_index": idx,
                    },
                )
            )
        buffer.clear()

    for line in page.content.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("==") and stripped.endswith("=="):
            flush()  # save whatever was buffered under the previous section
            section_name = stripped.strip("= ").strip()
            continue
        buffer.append(stripped)
        # flush every ~3 lines to keep chunks reasonably small
        if len(buffer) >= 3:
            flush()

    flush()  # save any trailing buffered content

    return documents
