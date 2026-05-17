import streamlit as st
import os
import logging

from src.utils.document_loader import process_and_get_chunks
from src.database.vector_store import db_manager
from src.models.rag import get_rag_chain
from src.config.settings import settings

logger = logging.getLogger("NexusAI.App")

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Nexus | GenAI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Session State
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "knowledge_base_ready" not in st.session_state:
    st.session_state.knowledge_base_ready = False


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.title("⚙️ Dashboard")

    st.subheader("🗄️ Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload Documents (TXT/PDF)",
        type=["txt", "pdf"],
        accept_multiple_files=True
    )

    if st.button("Sync & Initialize Vector Store"):
        with st.spinner("Processing documents..."):
            try:
                if uploaded_files:
                    os.makedirs(settings.DOCS_DIRECTORY, exist_ok=True)

                    for file in uploaded_files:
                        path = os.path.join(settings.DOCS_DIRECTORY, file.name)
                        with open(path, "wb") as f:
                            f.write(file.getbuffer())

                chunks = process_and_get_chunks()

                if not chunks:
                    st.warning("No documents found.")
                else:
                    db_manager.add_documents(chunks)
                    st.session_state.knowledge_base_ready = True
                    st.success(f"Indexed {len(chunks)} chunks!")

            except Exception as e:
                st.error(f"Error: {e}")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.caption(f"Model: {settings.LLM_MODEL_NAME}")


# =========================
# Header
# =========================
st.title("🤖 NEXUS AI")
st.caption("Advanced RAG Assistant")


# =========================
# Knowledge base check
# =========================
if not st.session_state.knowledge_base_ready:
    st.warning("Please initialize the knowledge base from sidebar.")
    st.stop()


# =========================
# Chat History
# =========================
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =========================
# Chat Input
# =========================
prompt = st.chat_input("Ask something...")

if prompt:
    # User message
    st.session_state.chat_history.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        try:
            rag = get_rag_chain()

            result = rag.stream_answer(prompt, st.session_state.chat_history)

            full_response = ""

            for chunk in result:
                if "text" in chunk:
                    full_response += chunk["text"]
                    st.write(chunk["text"])

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": full_response
            })

        except Exception as e:
            st.error(f"Error: {e}")


# =========================
# Footer
# =========================
st.markdown("---")
st.caption("Nexus AI © 2026 | Powered by LangChain + Groq")
