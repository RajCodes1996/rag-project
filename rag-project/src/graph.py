from langgraph.graph import END, StateGraph

import os
from langsmith import traceable

from src.llm import generate_answer
from src.rag_pipeline import TOP_K, RETRIEVAL_STRATEGY, get_pipeline
from src.retriever import format_context, retrieve_relevant_chunks
from src.state import GraphState


@traceable(name="retrieve_node", run_type="chain")
def retrieve_node(state: GraphState):
    question = state["question"]
    pipeline = get_pipeline()

    if not pipeline.is_ready or pipeline.vector_store is None:
        return {
            "retrieved_docs": [],
            "context": "No indexed documents are available yet.",
        }

    docs = retrieve_relevant_chunks(
        query=question,
        vector_store=pipeline.vector_store,
        top_k=TOP_K,
        strategy=RETRIEVAL_STRATEGY,
    )
    context = format_context(docs)

    return {
        "retrieved_docs": docs,
        "context": context,
    }


@traceable(name="generate_node", run_type="chain")
def generate_node(state: GraphState):
    question = state["question"]
    context = state["context"]

    if context == "No indexed documents are available yet.":
        return {"answer": context}

    answer = generate_answer(
        query=question,
        context=context,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    return {"answer": answer}


workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app_graph = workflow.compile()
