
from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings
from typing import cast

from app.config import get_settings
from app.ingestion.embeddings import get_embedding_model
from app.ingestion.document import Document

def get_vector_store()-> Chroma:
    settings=get_settings()
    return Chroma(
        collection_name=settings.vector_store_collection,
        embedding_function= cast(Embeddings, get_embedding_model()),
        persist_directory=settings.vector_store_path,
    )

def add_documents(documents:list [Document])->None:
    if not documents:
        return
    vector_store=get_vector_store()
    lc_docs=[
        LCDocument(page_content=doc.content,metadata=doc.metadata)
        for doc in documents
    ]
    vector_store.add_documents(lc_docs)

def similarity_search(query:str ,k:int=4, metadata_filter :  dict |None = None)->list[Document]:
    """
    Tool 2: Semantic Search + Tool 4: Metadata Filter (combined here since
    Chroma applies a metadata filter natively as part of the same query,
    rather than as a separate post-processing step).

    metadata_filter example: {"source_name": "paper.pdf"} restricts the
    search to chunks from just that document.
    """
    vector_store=get_vector_store()
    results=vector_store.similarity_search_with_relevance_scores(query,k=k, filter=metadata_filter)

    docs = []
    for doc, score in results:
        metadata = {**doc.metadata, "retrieval_method": "semantic", "relevance_score": float(score)}
        docs.append(Document(content=doc.page_content, metadata=metadata))
    return docs

def get_all_documents()->list[Document]:
    vector_store=get_vector_store()
    raw=vector_store.get(include=["documents","metadatas"])
    return[
       Document(content=content, metadata=metadata or {})
        for content, metadata in zip(raw["documents"], raw["metadatas"])
    ]
