
from pathlib import Path
from document_loader import load_document

test_folder = Path("test_docs")
test_folder.mkdir(exist_ok=True)

# Create a sample text document.
sample_file = test_folder / "ai_notes.txt"

sample_file.write_text(
    "Artificial Intelligence enables machines "
    "to perform tasks that normally require "
    "human intelligence. Machine learning is "
    "a subfield of artificial intelligence.",
    encoding="utf-8",
)

# Extract its content.
documents = load_document(str(sample_file))

for document in documents:
    print("Source:", document["source"])
    print("Page:", document["page"])
    print("Text:", document["text"])
