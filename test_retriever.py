
from retriever import retrieve_documents


question = "What is machine learning?"

results = retrieve_documents(question, k=4)

print("Question:", question)
print("Number of retrieved chunks:", len(results))

for i, chunk in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print("Source:", chunk["source"])
    print("Page:", chunk["page"])
    print("Chunk ID:", chunk["chunk_id"])
    print("Distance:", round(chunk["distance"], 4))
    print("Text:", chunk["text"])
