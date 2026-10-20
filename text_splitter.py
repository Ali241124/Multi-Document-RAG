
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[dict]:
    """
    Split documents into chunks while preserving source and page metadata.
    Chunk IDs increase across pages belonging to the same source.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    next_chunk_id = {}

    for document in documents:
        source = document["source"]
        text_chunks = splitter.split_text(document["text"])

        for chunk_text in text_chunks:
            chunk_id = next_chunk_id.get(source, 0)

            chunks.append({
                "text": chunk_text,
                "source": source,
                "page": document.get("page"),
                "chunk_id": chunk_id,
            })

            next_chunk_id[source] = chunk_id + 1

    return chunks
