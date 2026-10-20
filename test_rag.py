
from rag_pipeline import answer_question


question = "What is machine learning?"

result = answer_question(question)

print("\nAnswer:")
print(result["answer"])

print("\nSources:")

for source in result["sources"]:
    print(
        f"- {source['source']}, "
        f"page: {source['page']}, "
        f"chunk: {source['chunk_id']}"
    )
