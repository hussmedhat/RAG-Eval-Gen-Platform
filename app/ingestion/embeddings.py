#app/ingestion/embeddings.py

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import get_settings

def get_embedding_model()-> HuggingFaceEmbeddings:
    settings = get_settings()

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
    )
