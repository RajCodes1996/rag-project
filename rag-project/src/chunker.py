"""
chunker.py — Splits documents into overlapping chunks for better retrieval.

Key design decisions:
- chunk_size=800: Balances context richness vs. embedding quality.
- chunk_overlap=150: ~19% overlap prevents information loss at boundaries.
- Sentence-aware splitting: Prefers splitting on paragraphs > newlines > sentences.
"""

from typing import List

from src.langchain_compat import Document, RecursiveCharacterTextSplitter


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Split a list of documents into smaller overlapping chunks.

    Args:
        documents:     List of LangChain Document objects.
        chunk_size:    Target character count per chunk.
        chunk_overlap: Characters shared between consecutive chunks.

    Returns:
        List of chunk Documents, each preserving the parent's metadata.

    Raises:
        ValueError: If documents list is empty.
    """
    if not documents:
        raise ValueError("No documents provided for chunking.")

    splitter = RecursiveCharacterTextSplitter(
        # Prefer splitting at paragraph boundaries first, then sentences, then words.
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,   # records char offset in metadata
    )

    chunks = splitter.split_documents(documents)

    # Tag each chunk with its index so retrieval results are traceable
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    print(f"  Split {len(documents)} document(s) into {len(chunks)} chunk(s).")
    return chunks
