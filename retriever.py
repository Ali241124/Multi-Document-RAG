
from vector_store import load_vector_store


def retrieve_documents(
    question: str,
    k: int = 4,
    source: str | None = None,
    collection_name: str = "multi_document_rag",
) -> list[dict]:
    """Retrieve relevant chunks from the selected collection."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    vector_store = load_vector_store(collection_name)

    search_filter = {"source": source} if source else None

    results = vector_store.similarity_search_with_score(
        question,
        k=k,
        filter=search_filter,
    )

    return [
        {
            "text": document.page_content,
            "source": document.metadata.get("source", "Unknown"),
            "page": document.metadata.get("page"),
            "chunk_id": document.metadata.get("chunk_id"),
            "distance": float(distance),
        }
        for document, distance in results
    ]
