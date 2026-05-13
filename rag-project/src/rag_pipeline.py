"""
rag_pipeline.py - Orchestrates the full RAG workflow.

The pipeline loads documents, chunks them, builds or reloads a FAISS index,
and answers questions with Groq using retrieved context.
"""

import os
from pathlib import Path
from typing import Optional

from src.chunker import chunk_documents
from src.loader import load_documents
from src.llm import generate_answer
from src.retriever import format_context, retrieve_relevant_chunks
from src.vector_store import get_or_build_vector_store

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DOCS_DIR = "data/docs"
INDEX_PATH = "data/faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
RETRIEVAL_STRATEGY = "mmr"


class RAGPipeline:
    """
    Stateful RAG pipeline.
    """

    def __init__(self, rebuild: bool = False):
        self._vector_store = None
        self._ready = False
        self._build(rebuild)

    def _build(self, force_rebuild: bool = False) -> None:
        docs_path = Path(DOCS_DIR)

        if not docs_path.exists() or not any(docs_path.rglob("*")):
            print(
                f"  [RAGPipeline] No documents found in '{DOCS_DIR}'. "
                "Add files and restart the app."
            )
            return

        print("[RAGPipeline] Initialising ...")

        print("\n[1/3] Loading documents ...")
        documents = load_documents(DOCS_DIR)

        print("\n[2/3] Chunking documents ...")
        chunks = chunk_documents(
            documents,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        print("\n[3/3] Building vector store ...")
        self._vector_store = get_or_build_vector_store(
            chunks,
            index_path=INDEX_PATH,
            model_name=EMBEDDING_MODEL,
            force_rebuild=force_rebuild,
        )

        self._ready = True
        print("\n[RAGPipeline] Ready\n")

    def query(self, question: str) -> str:
        if not self._ready:
            return (
                "The pipeline is not ready yet.\n\n"
                f"Please add your documents to the `{DOCS_DIR}/` folder "
                "and restart the app so they can be indexed."
            )

        if not question.strip():
            return "Please enter a question."

        relevant_docs = retrieve_relevant_chunks(
            query=question,
            vector_store=self._vector_store,
            top_k=TOP_K,
            strategy=RETRIEVAL_STRATEGY,
        )

        if not relevant_docs:
            return (
                "I could not find relevant information in the documents for "
                "your question. Try rephrasing or uploading more relevant documents."
            )

        context = format_context(relevant_docs)
        api_key = os.getenv("GROQ_API_KEY")
        return generate_answer(
            query=question,
            context=context,
            api_key=api_key,
        )

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def vector_store(self):
        return self._vector_store


_pipeline_instance: Optional[RAGPipeline] = None


def get_pipeline(rebuild: bool = False) -> RAGPipeline:
    """Return the singleton RAGPipeline, building it on first call."""
    global _pipeline_instance
    if _pipeline_instance is None or rebuild:
        _pipeline_instance = RAGPipeline(rebuild=rebuild)
    return _pipeline_instance


def reset_pipeline() -> None:
    """Forget the cached pipeline instance so the next call rebuilds it."""
    global _pipeline_instance
    _pipeline_instance = None


def run_rag(query: str) -> str:
    pipeline = get_pipeline()
    return pipeline.query(query)
