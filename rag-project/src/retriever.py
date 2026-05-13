"""
retriever.py - Retrieval helpers for the RAG pipeline.

This module stays intentionally small and dependency-light so it can be used
from both the Streamlit app and the LangGraph workflow without creating
import cycles.
"""

from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langsmith import traceable

from src.langchain_compat import Document


@traceable(name="retrieve_relevant_chunks", run_type="retriever")
def retrieve_relevant_chunks(
    query: str,
    vector_store: FAISS,
    top_k: int = 5,
    strategy: str = "mmr",
    score_threshold: float = 0.25,
    fetch_k: int = 20,
    lambda_mult: float = 0.6,
) -> List[Document]:
    """
    Retrieve the most relevant document chunks for a query.
    """
    if not query.strip():
        raise ValueError("Query must not be empty.")

    if strategy not in {"mmr", "similarity"}:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'mmr' or 'similarity'.")

    if strategy == "mmr":
        return vector_store.max_marginal_relevance_search(
            query,
            k=top_k,
            fetch_k=max(fetch_k, top_k * 4),
            lambda_mult=lambda_mult,
        )

    docs_and_scores: List[Tuple[Document, float]] = (
        vector_store.similarity_search_with_relevance_scores(query, k=top_k * 2)
    )

    filtered = [doc for doc, score in docs_and_scores if score >= score_threshold]

    if not filtered:
        filtered = [doc for doc, _ in docs_and_scores[:top_k]]

    return filtered[:top_k]


def format_context(docs: List[Document]) -> str:
    """
    Format retrieved chunks into a readable context block for the prompt.
    """
    if not docs:
        return "No relevant context found."

    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_file", doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", "")
        page_str = f", page {page + 1}" if page != "" else ""
        header = f"[{i}] Source: {source}{page_str}"
        parts.append(f"{header}\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(parts)
