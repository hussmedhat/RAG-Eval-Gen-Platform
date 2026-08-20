import wikipedia

from app.ingestion.document import Document


def load_wikipedia(title: str, lang: str = "en") -> list[Document]:
    wikipedia.set_lang(lang)

    try:
        page = wikipedia.page(title, auto_suggest=False)
    except wikipedia.exceptions.DisambiguationError as e:
        # pick the first real option rather than failing outright
        page = wikipedia.page(e.options[0], auto_suggest=False)
    except wikipedia.exceptions.PageError:
        raise FileNotFoundError(f"Wikipedia page not found: {title}")

    documents: list[Document] = []

    # page.content contains "== Section ==" markers; split on those
    raw_sections = page.content.split("\n\n\n") if page.content else []
    section_name = "Introduction"

    for idx, chunk in enumerate(page.content.split("\n\n"), start=1):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.startswith("=="):
            section_name = chunk.strip("= ").strip()
            continue

        documents.append(
            Document(
                content=chunk,
                metadata={
                    "source_type": "wikipedia",
                    "source_name": page.title,
                    "url": page.url,
                    "section": section_name,
                    "section_index": idx,
                },
            )
        )

    return documents