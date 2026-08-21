#app/ingestion/vector_store.py

from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument

from app.config import get_settings
from app.ingestion.embeddings import get_embedding_model
from app.ingestion.document import Document


def get_vector_store()-> Chroma:
    settings=get_settings()
    return Chroma(
        collection_name=settings.vector_store_collection,
        embedding_function= get_embedding_model(),
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

def similarity_search(query:str ,k:int=4)->list[Document]:
    vector_store=get_vector_store()
    results=vector_store.similarity_search(query,k=k)

    return[
        Document(content=doc.page_content,metadata=doc.metadata)
        for doc in results
    ]
