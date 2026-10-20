
from document_loader import load_document
from text_splitter import split_documents

# Load the sample document created in Step 2.
documents = load_document("test_docs/ai_notes.txt")

# Split the extracted text into chunks.
chunks = split_documents(
    documents,
    chunk_size=100,
    chunk_overlap=20,
)

print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {i} ---")
    print("Source:", chunk["source"])
    print("Page:", chunk["page"])
    print("Chunk ID:", chunk["chunk_id"])
    print("Text:", chunk["text"])
