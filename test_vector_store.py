
from document_loader import load_document
from text_splitter import split_documents
from vector_store import create_vector_store


# 1. Load our sample document.
documents = load_document("test_docs/ai_notes.txt")

# 2. Split its text into chunks.
chunks = split_documents(
    documents,
    chunk_size=100,
    chunk_overlap=20,
)

print("Number of chunks:", len(chunks))

# 3. Create embeddings and store them in ChromaDB.
vector_store = create_vector_store(chunks)

print("\nVector database created successfully!")

# 4. Search using a natural-language question.
question = "What is machine learning?"

results = vector_store.similarity_search(
    question,
    k=2,
)

print("\nSearch results:")

for result in results:
    print("\nText:", result.page_content)
    print("Source:", result.metadata["source"])
    print("Chunk ID:", result.metadata["chunk_id"])
