"""
embeddings.py — Loads and caches a SentenceTransformer embedding model.

Model choice: 'all-MiniLM-L6-v2'
  - Small (80 MB), fast, and high quality for semantic similarity tasks.
  - 384-dimensional embeddings — ideal for FAISS cosine search.
  - Handles up to 256 tokens per chunk (our chunks are well within this).
"""

from functools import lru_cache
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings


# ---------------------------------------------------------------------------
# Singleton loader — model is downloaded once and reused across calls.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Return (and cache) a HuggingFace embedding model.

    Args:
        model_name: SentenceTransformer model identifier.

    Returns:
        HuggingFaceEmbeddings instance compatible with LangChain vector stores.
    """
    print(f"  Loading embedding model: {model_name} …")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,   # cosine similarity via dot product
            "batch_size": 32,
        },
    )
    print("  Embedding model ready.")
    return embeddings


def embed_texts(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    """
    Convenience helper — embed a list of plain strings.

    Args:
        texts:      Strings to embed.
        model_name: Model to use (default: all-MiniLM-L6-v2).

    Returns:
        List of embedding vectors.
    """
    model = get_embedding_model(model_name)
    return model.embed_documents(texts)
