
import tempfile
from pathlib import Path

import streamlit as st
import uuid

from document_loader import load_document
from text_splitter import split_documents
from vector_store import create_vector_store, load_vector_store
from rag_pipeline import answer_question


st.set_page_config(
    page_title="Multi-Document RAG",
    page_icon="📚",
    layout="wide",
)
if "collection_name" not in st.session_state:
    st.session_state.collection_name = (
        f"session_{uuid.uuid4().hex}"
    )
st.title("📚 Multi-Document RAG Assistant")
st.caption(
    "Ask questions about your documents and get answers "
    "grounded in retrieved passages."
)


# ---------------- SESSION STATE ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- HELPER FUNCTIONS ----------------

def get_indexed_sources(
    collection_name: str,
) -> list[str]:
    """Return filenames currently indexed in ChromaDB."""
    try:
        vector_store = load_vector_store(collection_name)
        results = vector_store._collection.get(
            include=["metadatas"]
        )

        return sorted({
            metadata["source"]
            for metadata in results["metadatas"]
            if metadata and metadata.get("source")
        })

    except FileNotFoundError:
        return []


def display_sources(sources: list[dict]) -> None:
    """Display citations and expandable retrieved passages."""
    if not sources:
        st.caption("No source passages were returned.")
        return

    with st.expander("View sources and retrieved passages"):
        for index, source in enumerate(sources, start=1):
            location = source["source"]

            if source.get("page") is not None:
                location += f" — Page {source['page']}"

            st.markdown(f"**{index}. {location}**")

            if source.get("chunk_id") is not None:
                st.caption(f"Chunk ID: {source['chunk_id']}")

            if source.get("text"):
                st.text(source["text"])

            if index < len(sources):
                st.divider()


# ---------------- DOCUMENT UPLOAD ----------------

with st.expander("1. Upload and process documents", expanded=True):
    uploaded_files = st.file_uploader(
        "Choose PDF, TXT, or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        key="document_uploader",
    )

    if st.button("Process documents", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            all_chunks = []
            processed_names = []
            failed_files = []

            with st.spinner(
                "Reading documents and creating embeddings..."
            ):
                with tempfile.TemporaryDirectory() as temp_dir:
                    for uploaded_file in uploaded_files:
                        safe_name = Path(uploaded_file.name).name
                        file_path = Path(temp_dir) / safe_name

                        try:
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
                            else:
                                failed_files.append(
                                    f"{safe_name}: no readable text"
                                )

                        except Exception as error:
                            failed_files.append(
                                f"{safe_name}: {error}"
                            )

                if all_chunks:
                    try:
                        create_vector_store(
    all_chunks,
    collection_name=st.session_state.collection_name,
)

                        st.success(
                            f"Processed {len(processed_names)} file(s) "
                            f"and indexed {len(all_chunks)} chunks."
                        )
                    except Exception as error:
                        st.error(
                            f"Could not update the vector database: {error}"
                        )

                elif not failed_files:
                    st.error(
                        "No readable text was found in the uploaded files."
                    )

            if failed_files:
                st.warning("Some files could not be processed:")
                for failure in failed_files:
                    st.write(f"- {failure}")


# ---------------- INDEXED DOCUMENTS ----------------

st.header("2. Indexed documents")

indexed_sources = get_indexed_sources(
    st.session_state.collection_name
)

if indexed_sources:
    st.caption(f"{len(indexed_sources)} document(s) indexed")

    for filename in indexed_sources:
        st.write(f"📄 {filename}")
else:
    st.info("No indexed documents found. Upload and process documents first.")


# ---------------- DOCUMENT FILTER ----------------

st.header("3. Ask questions")

source_options = ["All documents"] + indexed_sources

selected_source = st.selectbox(
    "Search in",
    options=source_options,
    help="Choose one document or search across all indexed documents.",
)

source_override = (
    None if selected_source == "All documents"
    else selected_source
)


# ---------------- CHAT HISTORY ----------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            display_sources(message["sources"])


# ---------------- QUESTION INPUT ----------------

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
                result = answer_question(
                    question,
                    source_override=source_override,
                    collection_name=st.session_state.collection_name,
                )

            answer = result["answer"]
            sources = result["sources"]

            st.markdown(answer)
            display_sources(sources)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        except Exception as error:
            st.error(f"Unable to answer the question: {error}")
