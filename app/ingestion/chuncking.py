#app/ingestion/chuncking.py

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.document import Document

def chunk_documents(
        documents: list[Document],
        chunk_size: int =500,
        chunk_overlap: int=100,
)-> list[Document]:
    
    splitter= RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunked_documents: list[Document] = []
    for doc in documents:
        splits = splitter.split_text(doc.content)

        for i, chunk in enumerate(splits):
            new_metadata = dict(doc.metadata)
            new_metadata["chunk_index"] = i
            new_metadata["total_chunks"] = len(splits)

            chunked_documents.append(Document(content=chunk, metadata=new_metadata))

    return chunked_documents

