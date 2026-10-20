
import streamlit as st
import tempfile
from pathlib import Path

from document_loader import load_document
from text_splitter import split_documents
from vector_store import create_vector_store
from rag_pipeline import answer_question


st.set_page_config(
    page_title="Multi-Document RAG",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Multi-Document RAG Assistant")

st.write(
    "Upload documents, ask questions, and get answers "
    "grounded in your files."
)


# Keep track of uploaded files in this browser session.
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- DOCUMENT UPLOAD ----------------

st.header("1. Upload your documents")

uploaded_files = st.file_uploader(
    "Choose PDF, TXT, or DOCX files",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True,
)

if st.button("Process documents", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one document.")
    else:
        all_chunks = []
        processed_names = []

        try:
            with st.spinner(
                "Reading documents and creating embeddings..."
            ):
                with tempfile.TemporaryDirectory() as temp_dir:
                    for uploaded_file in uploaded_files:
                        # Use only the filename, not a user-supplied path.
                        safe_name = Path(
                            uploaded_file.name
                        ).name

                        file_path = (
                            Path(temp_dir) / safe_name
                        )

                        file_path.write_bytes(
                            uploaded_file.getvalue()
                        )

                        documents = load_document(
                            str(file_path)
                        )

                        chunks = split_documents(documents)

                        if chunks:
                            all_chunks.extend(chunks)
                            processed_names.append(safe_name)

                if not all_chunks:
                    st.error(
                        "No readable text was found in the "
                        "uploaded documents."
                    )
                else:
                    create_vector_store(all_chunks)

                    st.session_state.uploaded_files = sorted(
                        set(
                            st.session_state.uploaded_files
                            + processed_names
                        )
                    )

                    st.success(
                        f"Processed {len(processed_names)} file(s) "
                        f"and indexed {len(all_chunks)} chunks."
                    )

        except Exception as error:
            st.error(
                f"Document processing failed: {error}"
            )


# ---------------- DOCUMENT LIST ----------------

st.header("2. Indexed documents")

if st.session_state.uploaded_files:
    for filename in st.session_state.uploaded_files:
        st.write(f"📄 {filename}")
else:
    st.caption(
        "Upload and process documents to begin."
    )


# ---------------- CHAT INTERFACE ----------------

st.header("3. Ask questions")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View retrieved sources"):
                for source in message["sources"]:
                    location = source["source"]

                    if source.get("page") is not None:
                        location += f", page {source['page']}"

                    st.write(f"- {location}")


question = st.chat_input(
    "Ask a question about your documents..."
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner(
                "Searching documents and generating an answer..."
            ):
                result = answer_question(question)

            answer = result["answer"]
            sources = result["sources"]

            st.markdown(answer)

            if sources:
                with st.expander("View retrieved sources"):
                    for source in sources:
                        location = source["source"]

                        if source.get("page") is not None:
                            location += (
                                f", page {source['page']}"
                            )

                        st.write(f"- {location}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        except Exception as error:
            st.error(f"Unable to answer the question: {error}")
