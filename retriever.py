
from vector_store import load_vector_store


def retrieve_documents(
    question: str,
    k: int = 4,
    source: str | None = None,
) -> list[dict]:
    """
    Retrieve relevant chunks from ChromaDB.

    If source is provided, search only that document.
    Otherwise, search across all indexed documents.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    vector_store = load_vector_store()

    # Restrict retrieval to a specific source when requested.
    search_filter = None

    if source:
        search_filter = {"source": source}

    results = vector_store.similarity_search_with_score(
        question,
        k=k,
        filter=search_filter,
    )

    retrieved_chunks = []

    for document, distance in results:
        retrieved_chunks.append({
            "text": document.page_content,
            "source": document.metadata.get(
                "source", "Unknown"
            ),
            "page": document.metadata.get("page"),
            "chunk_id": document.metadata.get("chunk_id"),
            "distance": float(distance),
        })

    return retrieved_chunks
