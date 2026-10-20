
import tempfile
from pathlib import Path

import streamlit as st

from chat_store import (
    initialize_db,
    create_chat,
    list_chats,
    get_chat,
    get_messages,
    save_message,
    rename_chat,
    delete_chat,
)
from document_loader import load_document
from text_splitter import split_documents
from vector_store import create_vector_store, load_vector_store
from rag_pipeline import answer_question


st.set_page_config(
    page_title="Multi-Document RAG",
    page_icon="📚",
    layout="wide",
)

initialize_db()


# ---------------- CHAT MANAGEMENT ----------------

if "active_chat_id" not in st.session_state:
    existing_chats = list_chats()

    if existing_chats:
        st.session_state.active_chat_id = existing_chats[0]["id"]
    else:
        new_chat = create_chat()
        st.session_state.active_chat_id = new_chat["id"]


active_chat = get_chat(st.session_state.active_chat_id)

if active_chat is None:
    new_chat = create_chat()
    st.session_state.active_chat_id = new_chat["id"]
    active_chat = new_chat


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("📚 RAG Assistant")

    if st.button("＋ New chat", use_container_width=True, type="primary"):
        new_chat = create_chat()
        st.session_state.active_chat_id = new_chat["id"]
        st.rerun()

    st.divider()
    st.caption("RECENT CHATS")

    saved_chats = list_chats()

    if not saved_chats:
        st.caption("Your conversations will appear here.")
    else:
        for chat in saved_chats:
            col_chat, col_delete = st.columns([5, 1])

            with col_chat:
                label = chat["title"][:32]

                if st.button(
                    label,
                    key=f"open_{chat['id']}",
                    use_container_width=True,
                    type=(
                        "primary"
                        if chat["id"] == st.session_state.active_chat_id
                        else "secondary"
                    ),
                ):
                    st.session_state.active_chat_id = chat["id"]
                    st.rerun()

            with col_delete:
                if st.button(
                    "×",
                    key=f"delete_{chat['id']}",
                    help="Delete this conversation",
                ):
                    delete_chat(chat["id"])

                    if chat["id"] == st.session_state.active_chat_id:
                        remaining = list_chats()

                        if remaining:
                            st.session_state.active_chat_id = remaining[0]["id"]
                        else:
                            new_chat = create_chat()
                            st.session_state.active_chat_id = new_chat["id"]

                    st.rerun()


# Refresh the selected conversation after sidebar actions.
active_chat = get_chat(st.session_state.active_chat_id)
messages = get_messages(active_chat["id"])


# ---------------- MAIN PAGE ----------------

st.title("Multi-Document RAG")
st.caption("Chat with your own documents. Your conversations are saved locally.")


# ---------------- DOCUMENT UPLOAD ----------------

with st.expander("Upload documents to this chat", expanded=not bool(messages)):
    uploaded_files = st.file_uploader(
        "Choose PDF, TXT, or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        key=f"upload_{active_chat['id']}",
    )

    if st.button(
        "Process documents",
        key=f"process_{active_chat['id']}",
        type="primary",
    ):
        if not uploaded_files:
            st.warning("Please choose at least one document.")
        else:
            all_chunks = []
            processed_names = []
            failed_files = []

            with st.spinner("Reading documents and creating embeddings..."):
                with tempfile.TemporaryDirectory() as temp_dir:
                    for uploaded_file in uploaded_files:
                        safe_name = Path(uploaded_file.name).name
                        file_path = Path(temp_dir) / safe_name

                        try:
                            file_path.write_bytes(uploaded_file.getvalue())

                            documents = load_document(str(file_path))
                            chunks = split_documents(documents)

                            if chunks:
                                all_chunks.extend(chunks)
                                processed_names.append(safe_name)
                            else:
                                failed_files.append(
                                    f"{safe_name}: no readable text"
                                )

                        except Exception as error:
                            failed_files.append(f"{safe_name}: {error}")

                if all_chunks:
                    try:
                        create_vector_store(
                            all_chunks,
                            collection_name=active_chat["collection_name"],
                        )

                        st.success(
                            f"Indexed {len(all_chunks)} chunks from "
                            f"{len(processed_names)} file(s)."
                        )
                    except Exception as error:
                        st.error(f"Could not index documents: {error}")

                if failed_files:
                    st.warning("Some documents could not be processed.")
                    for failure in failed_files:
                        st.write(f"- {failure}")


# ---------------- DOCUMENTS FOR THIS CHAT ----------------

st.subheader("Documents in this chat")

try:
    vector_store = load_vector_store(active_chat["collection_name"])

    results = vector_store._collection.get(include=["metadatas"])

    indexed_sources = sorted({
        metadata["source"]
        for metadata in results["metadatas"]
        if metadata and metadata.get("source")
    })
except Exception:
    indexed_sources = []

if indexed_sources:
    for filename in indexed_sources:
        st.write(f"📄 {filename}")
else:
    st.caption("No documents uploaded to this chat yet.")


# ---------------- SEARCH SCOPE ----------------

source_options = ["All documents"] + indexed_sources

selected_source = st.selectbox(
    "Search in",
    options=source_options,
    key=f"source_{active_chat['id']}",
)

source_override = (
    None if selected_source == "All documents"
    else selected_source
)


# ---------------- DISPLAY SAVED MESSAGES ----------------

if not messages:
    st.markdown("### What would you like to learn?")
    st.write("Upload a document above, then ask a question about its contents.")

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View sources and retrieved passages"):
                for source in message["sources"]:
                    location = source["source"]

                    if source.get("page") is not None:
                        location += f" — Page {source['page']}"

                    st.markdown(f"**{location}**")

                    if source.get("text"):
                        st.text(source["text"])

                    st.divider()


# ---------------- CHAT INPUT ----------------

question = st.chat_input("Ask about your documents...")

if question:
    save_message(
        active_chat["id"],
        "user",
        question,
    )

    if active_chat["title"] == "New Chat":
        rename_chat(active_chat["id"], question[:40])

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your documents and generating an answer..."):
                result = answer_question(
                    question,
                    source_override=source_override,
                    collection_name=active_chat["collection_name"],
                )

            answer = result["answer"]
            sources = result["sources"]

            st.markdown(answer)

            if sources:
                with st.expander("View sources and retrieved passages"):
                    for source in sources:
                        location = source["source"]

                        if source.get("page") is not None:
                            location += f" — Page {source['page']}"

                        st.markdown(f"**{location}**")

                        if source.get("text"):
                            st.text(source["text"])

                        st.divider()

            save_message(
                active_chat["id"],
                "assistant",
                answer,
                sources=sources,
            )

        except Exception as error:
            st.error(f"Unable to answer the question: {error}")
