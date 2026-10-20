
from pathlib import Path
import hashlib
import streamlit as st

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


DB_DIRECTORY = "chroma_db"
COLLECTION_NAME = "multi_document_rag"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource
def get_embedding_model():
    """Load and cache the embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


def load_vector_store(collection_name: str = COLLECTION_NAME):
    """Open a specific user's vector collection."""
    if not Path(DB_DIRECTORY).exists():
        raise FileNotFoundError(
            "Database not found. Upload documents first."
        )

    return Chroma(
        persist_directory=DB_DIRECTORY,
        embedding_function=get_embedding_model(),
        collection_name=collection_name,
    )


def create_vector_store(
    chunks: list[dict],
    collection_name: str = COLLECTION_NAME,
):
    """Replace old chunks for uploaded sources and preserve other sources."""
    if not chunks:
        raise ValueError("No text chunks were provided.")

    embeddings = get_embedding_model()

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=DB_DIRECTORY,
    )

    # Find each source being uploaded.
    sources = {chunk["source"] for chunk in chunks}

    # Remove its previous chunks before adding the updated version.
    for source in sources:
        vector_store.delete(
            where={"source": source}
        )

    documents = []
    ids = []

    for chunk in chunks:
        metadata = {
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
        }

        if chunk.get("page") is not None:
            metadata["page"] = chunk["page"]

        documents.append(
            Document(
                page_content=chunk["text"],
                metadata=metadata,
            )
        )

        unique_text = (
            f"{chunk['source']}|"
            f"{chunk.get('page')}|"
            f"{chunk['chunk_id']}|"
            f"{chunk['text']}"
        )

        ids.append(
            hashlib.sha256(
                unique_text.encode("utf-8")
            ).hexdigest()
        )

    vector_store.add_documents(
        documents=documents,
        ids=ids,
    )

    return vector_store
