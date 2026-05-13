# RAG Project

Retrieval-Augmented Generation (RAG) system for document-based question answering.

## Project Structure

```
rag-project/
├── data/
│   └── docs/          # Directory for storing documents
├── src/
│   ├── loader.py      # Document loading functionality
│   ├── chunker.py     # Document chunking and splitting
│   ├── embeddings.py  # Embedding generation
│   ├── vector_store.py # Vector storage and management
│   ├── retriever.py   # Document retrieval logic
│   ├── llm.py         # LLM interaction module
│   └── rag_pipeline.py # Main RAG pipeline orchestration
├── app.py             # Application entry point
├── requirements.txt   # Project dependencies
└── README.md          # This file
```

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place your documents in `data/docs/`

3. Run the application:
   ```bash
   python app.py
   ```

## Module Descriptions

- **loader.py**: Handles loading documents from various formats
- **chunker.py**: Splits documents into chunks for processing
- **embeddings.py**: Generates vector embeddings
- **vector_store.py**: Manages vector database operations
- **retriever.py**: Retrieves relevant documents
- **llm.py**: Manages language model interactions
- **rag_pipeline.py**: Orchestrates the complete RAG workflow
