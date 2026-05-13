from typing import List, TypedDict

from src.langchain_compat import Document

class GraphState(TypedDict):
    question: str
    retrieved_docs: List[Document]
    context: str
    answer: str
