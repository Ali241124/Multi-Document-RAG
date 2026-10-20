
from pathlib import Path
import re
import pymupdf
from docx import Document


def clean_text(text: str) -> str:
    """Clean common PDF text artifacts without removing paragraph breaks."""

    # Remove common model/token artifacts if they appear in extracted text.
    text = re.sub(r"<(?:pad|EOS)>", " ", text, flags=re.IGNORECASE)

    # Normalize Windows and old-style line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove spaces and tabs at the start/end of each line.
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Reduce excessive blank lines while preserving paragraph separation.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalize repeated spaces and tabs.
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def load_document(file_path: str) -> list[dict]:
    """
    Extract text from a PDF, TXT, or DOCX file.
    Return text with source and page metadata.
    """

    path = Path(file_path)
    extension = path.suffix.lower()

    if extension == ".pdf":
        pages = []

        with pymupdf.open(path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = clean_text(page.get_text("text"))

                if text:
                    pages.append({
                        "text": text,
                        "source": path.name,
                        "page": page_number,
                    })

        return pages

    elif extension == ".txt":
        text = clean_text(path.read_text(encoding="utf-8"))

        return [{
            "text": text,
            "source": path.name,
            "page": None,
        }]

    elif extension == ".docx":
        document = Document(path)

        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        )

        text = clean_text(text)

        return [{
            "text": text,
            "source": path.name,
            "page": None,
        }]

    else:
        raise ValueError(f"Unsupported file type: {extension}")
