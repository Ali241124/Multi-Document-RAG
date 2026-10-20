
from retriever import retrieve_documents

questions = [
    "What problem does Attention Is All You Need address?",
    "Why did the authors propose the Transformer architecture?",
    "What are the main components of the Transformer model?",
]

for question in questions:
    print("\n" + "=" * 70)
    print("QUESTION:", question)
    print("=" * 70)

    results = retrieve_documents(question, k=6)

    if not results:
        print("No passages retrieved.")
        continue

    for i, result in enumerate(results, start=1):
        print(f"\n--- Retrieved passage {i} ---")
        print("Source:", result["source"])
        print("Page:", result["page"])
        print("Distance:", round(result["distance"], 4))
        print("Text:", result["text"][:1000])
