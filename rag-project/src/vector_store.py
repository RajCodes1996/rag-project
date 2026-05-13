"""
vector_store.py — Builds, persists, and loads a FAISS vector store.

Why FAISS?
  - In-process, no server needed.
  - Supports L2 and inner-product (cosine after normalisation) search.
  - Scales to millions of vectors on a single machine.

Persistence strategy:
  - Index is saved to disk after building so subsequent app starts skip
    the expensive embed-and-index step.
  - Index is invalidated (re-built) when new documents are added.
"""

import os
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS

from src.embeddings import get_embedding_model
from src.langchain_compat import Document

# Default path where the FAISS index is persisted
DEFAULT_INDEX_PATH = "data/faiss_index"


def build_vector_store(
    chunks: List[Document],
    index_path: str = DEFAULT_INDEX_PATH,
    model_name: str = "all-MiniLM-L6-v2",
) -> FAISS:
    """
    Embed chunks and create a new FAISS index, then persist it to disk.

    Args:
        chunks:     Document chunks to index.
        index_path: Directory where the index files will be saved.
        model_name: Embedding model to use.

    Returns:
        FAISS vector store instance.

    Raises:
        ValueError: If chunks list is empty.
    """
    if not chunks:
        raise ValueError("Cannot build vector store from an empty chunk list.")

    embeddings = get_embedding_model(model_name)
    print(f"  Embedding {len(chunks)} chunks and building FAISS index …")

    vector_store = FAISS.from_documents(chunks, embeddings)

    # Persist to disk so we don't re-embed on every app restart
    Path(index_path).mkdir(parents=True, exist_ok=True)
    vector_store.save_local(index_path)
    print(f"  FAISS index saved to: {index_path}")

    return vector_store


def load_vector_store(
    index_path: str = DEFAULT_INDEX_PATH,
    model_name: str = "all-MiniLM-L6-v2",
) -> Optional[FAISS]:
    """
    Load a previously saved FAISS index from disk.

    Args:
        index_path: Directory containing the saved index files.
        model_name: Must match the model used when building the index.

    Returns:
        FAISS instance, or None if no saved index is found.
    """
    index_dir = Path(index_path)
    index_file = index_dir / "index.faiss"

    if not index_file.exists():
        return None

    embeddings = get_embedding_model(model_name)
    print(f"  Loading existing FAISS index from: {index_path}")
    return FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,  # safe because we wrote the index ourselves
    )


def get_or_build_vector_store(
    chunks: List[Document],
    index_path: str = DEFAULT_INDEX_PATH,
    model_name: str = "all-MiniLM-L6-v2",
    force_rebuild: bool = False,
) -> FAISS:
    """
    Return a cached index if it exists, otherwise build a new one.

    Args:
        chunks:        Chunks to use when building from scratch.
        index_path:    Persistent index directory.
        model_name:    Embedding model identifier.
        force_rebuild: If True, always re-embed and rebuild.

    Returns:
        FAISS vector store ready for similarity search.
    """
    if not force_rebuild:
        existing = load_vector_store(index_path, model_name)
        if existing is not None:
            return existing

    return build_vector_store(chunks, index_path, model_name)
