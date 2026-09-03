"""
Tool 2 : Keyword Search  (BM25)

This tool matches by exact/lexical overlap — critical for things like
model names, IDs, or exact phrases that an embedding model might blur
together with similar-but-wrong text.
"""

from rank_bm25 import BM25Okapi
from app.ingestion.document import Document
from app.knowledge_store.vector_store import get_all_documents

def _tokenize(text : str) -> list[str]:
    """Tokenizes the input text into a list of lowercase words."""
    return text.lower().split()

def keyword_search(query : str , k : int = 4) -> list[Document]:
    """Returns the top-k chunks ranked by BM25 lexical overlap with the query."""
    all_documents = get_all_documents()
    if not all_documents:
        return []
    tokenized_corpus = [_tokenize(doc.content) for doc in all_documents]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize(query))
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_docs = []
    for i in ranked_indices[:k]:
        if scores[i] < 0:
            break
        doc = all_documents[i]
        doc.metadata = {**doc.metadata,"retrieval_method" : "keyword","bm25_score" : scores[i]}
        top_docs.append(doc)
    return top_docs
