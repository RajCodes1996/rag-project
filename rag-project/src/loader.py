"""
loader.py — Document loading from PDFs, TXTs, and Markdown files.
Supports loading from a directory or a single file path.
"""

import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from src.langchain_compat import Document


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_documents(source: str) -> List[Document]:
    """
    Load documents from a file or directory.

    Args:
        source: Path to a file or directory containing documents.

    Returns:
        List of LangChain Document objects.

    Raises:
        ValueError: If source path doesn't exist or has no supported files.
    """
    source_path = Path(source)

    if not source_path.exists():
        raise ValueError(f"Source path does not exist: {source}")

    if source_path.is_file():
        return _load_single_file(source_path)

    if source_path.is_dir():
        return _load_directory(source_path)

    raise ValueError(f"Source must be a file or directory: {source}")


def _load_single_file(path: Path) -> List[Document]:
    """Load a single document by file type."""
    ext = path.suffix.lower()

    if ext == ".pdf":
        loader = PyPDFLoader(str(path))
    elif ext in {".txt", ".md"}:
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}")

    docs = loader.load()

    # Enrich metadata
    for doc in docs:
        doc.metadata["source_file"] = path.name
        doc.metadata["file_type"] = ext

    return docs


def _load_directory(directory: Path) -> List[Document]:
    """Load all supported documents from a directory recursively."""
    all_docs: List[Document] = []

    files = [
        f for f in directory.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        raise ValueError(
            f"No supported documents found in '{directory}'. "
            f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    for file_path in sorted(files):
        try:
            docs = _load_single_file(file_path)
            all_docs.extend(docs)
            print(f"  Loaded: {file_path.name} ({len(docs)} page(s))")
        except Exception as e:
            print(f"  Skipped {file_path.name}: {e}")

    print(f"\n  Total documents loaded: {len(all_docs)}")
    return all_docs
