
# Multi-Document RAG Assistant

A document question-answering application powered by Retrieval-Augmented Generation (RAG). Upload PDF, TXT, and DOCX files, ask questions about their content, and receive AI-generated answers grounded in retrieved document passages.

The application combines LangChain, ChromaDB, Hugging Face embeddings, and a locally running Ollama language model. It also supports saved conversations and separate document collections for individual chats.

## Features

- **Multi-format document support:** Upload PDF, TXT, and DOCX files.
- **Semantic search:** Retrieve relevant passages using vector embeddings.
- **AI-powered question answering:** Generate answers using document context and a local language model.
- **Source references:** Display source filenames, page numbers when available, and retrieved passages.
- **Document-specific search:** Search all documents in a chat or select a particular document.
- **Chat history:** Save conversations locally and revisit previous chats.
- **New chat support:** Start separate conversations with independent document collections.
- **Persistent vector storage:** Store document embeddings in ChromaDB.
- **Local inference:** Use Ollama to run the language model on your own machine.

## Tech Stack

| Component | Technology |
|---|---|
| User interface | Streamlit |
| Programming language | Python |
| RAG framework | LangChain |
| Vector database | ChromaDB |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Language model | Qwen 2.5 7B via Ollama |
| Chat persistence | SQLite |
| Document processing | PyMuPDF, python-docx |

## Architecture

1. **Document ingestion:** The user uploads supported documents.
2. **Text extraction:** Document loaders extract text and available metadata.
3. **Text chunking:** The extracted text is split into smaller passages.
4. **Embedding generation:** Hugging Face embeddings convert passages into vectors.
5. **Vector storage:** ChromaDB stores the vectors, text, and source metadata.
6. **Retrieval:** Relevant passages are retrieved for a user's question.
7. **Answer generation:** Ollama generates an answer using the retrieved context.
8. **Source display:** The application displays source references and retrieved passages.
9. **Chat persistence:** SQLite stores conversation messages, titles, and source references.

## Project Structure

```text
Multi-Document-RAG/
├── app.py
├── chat_store.py
├── document_loader.py
├── text_splitter.py
├── vector_store.py
├── retriever.py
├── rag_pipeline.py
├── requirements.txt
├── .gitignore
├── test_docs/
│   └── ai_notes.txt
├── test_chunking.py
├── test_loader.py
├── test_rag.py
├── test_retriever.py
└── test_vector_store.py
```

Local data directories and database files may be created when you run the application.

## Prerequisites

Before getting started, install:

- Python 3.10 or a compatible version supported by your dependencies.
- Git.
- Ollama.
- The Qwen 2.5 7B model.

Check your Python and Git installations:

```bash
python --version
git --version
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ali241124/Multi-Document-RAG.git
cd Multi-Document-RAG
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, consult the official Python or PowerShell documentation for your environment's execution policy.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install and start Ollama

Install Ollama from:

https://ollama.com/

Download the model:

```powershell
ollama pull qwen2.5:7b
```

Make sure Ollama is running before asking the application questions. Depending on your installation, Ollama may already run as a background service.

### 5. Run the application

```powershell
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

http://localhost:8501

## How to Use

1. Launch the application.
2. Create a new chat.
3. Upload one or more PDF, TXT, or DOCX files.
4. Click **Process documents**.
5. Select a specific document or search across all documents in the current chat.
6. Ask a question using the chat input.
7. Review the generated answer and expand the source section to inspect retrieved passages.
8. Return to earlier conversations through the recent chats sidebar.

## Configuration

The current RAG pipeline uses the following models:

**Embedding model**

```text
sentence-transformers/all-MiniLM-L6-v2
```

**Language model**

```text
qwen2.5:7b
```

The embedding model is cached through Streamlit's resource cache to avoid unnecessary repeated initialization.

## Testing

Run the project's test scripts individually:

```powershell
python test_loader.py
python test_chunking.py
python test_vector_store.py
python test_retriever.py
python test_rag.py
```

Some tests may require existing indexed documents, a running Ollama instance, or previously downloaded models.

## Data Storage and Privacy

- ChromaDB stores indexed document passages and embeddings locally.
- SQLite stores saved chat messages, titles, and source references.
- The original uploaded files are not permanently archived by the current upload workflow.
- Local database files should not be committed to a public repository.
- The current application is intended for local development and does not yet provide user authentication or production-grade multi-user access controls.

Do not upload confidential documents or expose private chat data when deploying a shared instance.

## Current Limitations

- Answer quality depends on retrieval relevance and the information present in the uploaded documents.
- Scanned PDFs may require OCR support if they contain no extractable text.
- Local inference performance depends on available hardware and memory.
- Chat and vector database cleanup should be coordinated when conversations are deleted.
- Public deployment requires additional work for authentication, storage persistence, and per-user data isolation.

## Future Improvements

- Improve retrieval quality with reranking and hybrid search.
- Add automated tests for unsupported questions and source-grounded answers.
- Add chat renaming and complete document-management controls.
- Improve conversation deletion and vector collection cleanup.
- Add authentication and secure per-user storage.
- Support cloud deployment with a hosted language model API.

## Author

**Syed Ali Hassan**

- GitHub: https://github.com/Ali241124
- Portfolio: https://syedalihassan-portfolio.vercel.app/

---

If you find this project useful, feel free to explore the repository and suggest improvements.
