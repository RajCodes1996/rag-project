"""Compatibility helpers for LangChain import paths.

LangChain 1.x moved several commonly used symbols out of the old
``langchain`` namespace. The project keeps a small compatibility layer so it
works with both the newer package layout and older 0.3-style installs.
"""

from __future__ import annotations

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover - fallback for older LangChain installs
    from langchain.schema import Document  # type: ignore

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - fallback for older LangChain installs
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

