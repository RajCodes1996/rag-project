"""
app.py - Streamlit UI for the RAG Research Assistant.
"""

import os
from pathlib import Path

import streamlit as st
from langsmith import traceable

from src.graph import app_graph
from src.langsmith_config import configure_langsmith_env
from src.llm import normalize_api_key
from src.rag_pipeline import DOCS_DIR, get_pipeline, reset_pipeline
from src.langsmith_smoke_test import run_smoke_test


configure_langsmith_env()


st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="🔍",
    layout="wide",
)

DOCS_PATH = Path(DOCS_DIR)
DOCS_PATH.mkdir(parents=True, exist_ok=True)

GROQ_API_KEY = normalize_api_key(os.getenv("GROQ_API_KEY", "")) or ""


@st.cache_resource(show_spinner=False)
def load_pipeline():
    return get_pipeline()


@traceable(name="answer_user_question", run_type="chain")
def answer_user_question(prompt: str) -> str:
    """Run the LangGraph answer flow under one top-level LangSmith trace."""
    result = app_graph.invoke({"question": prompt})
    return result["answer"]


with st.sidebar:
    st.title("⚙️ Settings")

    api_key_input = st.text_input(
        "Groq API Key",
        value=GROQ_API_KEY,
        type="password",
        help="Get your key at console.groq.com",
    )
    if api_key_input:
        cleaned_key = normalize_api_key(api_key_input)
        if cleaned_key:
            os.environ["GROQ_API_KEY"] = cleaned_key

    st.divider()

    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop PDFs, TXT, or Markdown files here",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        new_files_saved = False
        for uploaded_file in uploaded_files:
            destination = DOCS_PATH / uploaded_file.name
            if not destination.exists():
                destination.write_bytes(uploaded_file.read())
                st.success(f"Saved: {uploaded_file.name}")
                new_files_saved = True
            else:
                st.info(f"Already indexed: {uploaded_file.name}")

        if new_files_saved:
            st.warning("New files detected. Click Rebuild Index below.")

    existing = sorted(DOCS_PATH.glob("*"))
    if existing:
        st.subheader("📚 Indexed Documents")
        for file_path in existing:
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"• `{file_path.name}`")
            if col2.button("🗑", key=f"del_{file_path.name}", help=f"Remove {file_path.name}"):
                file_path.unlink()
                st.rerun()

    st.divider()

    if st.button("🔄 Rebuild Index", use_container_width=True):
        reset_pipeline()
        st.cache_resource.clear()
        st.rerun()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

    if st.button("🧪 Test LangSmith", use_container_width=True):
        try:
            with st.spinner("Sending a test trace to LangSmith ..."):
                result = run_smoke_test("LangSmith sidebar test from Streamlit")
            st.success("LangSmith test trace sent.")
            st.code(result, language="json")
        except Exception as exc:
            st.error(f"LangSmith test failed: {exc}")


st.title("🔍 RAG Research Assistant")
st.caption("Ask questions about your uploaded documents. Powered by Groq + FAISS + LangChain.")

with st.spinner("Initialising RAG pipeline ..."):
    pipeline = load_pipeline()

if not pipeline.is_ready:
    st.warning(
        "📂 No documents are indexed yet. Use the sidebar to upload PDF, TXT, "
        "or Markdown files, then rebuild the index."
    )
else:
    st.success(f"✅ Pipeline ready - {len(list(DOCS_PATH.glob('*')))} document(s) indexed.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents ..."):
    if not normalize_api_key(os.environ.get("GROQ_API_KEY")):
        st.error("Please enter your Groq API key in the sidebar first.")
        st.stop()

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer ..."):
            try:
                answer = answer_user_question(prompt)
            except Exception as exc:
                answer = f"Error: {exc}"
        st.markdown(answer)

    st.session_state["messages"].append({"role": "assistant", "content": answer})
