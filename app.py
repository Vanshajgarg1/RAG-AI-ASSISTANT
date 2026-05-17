import streamlit as st
import os
import logging
from src.utils.document_loader import process_and_get_chunks
from src.database.vector_store import db_manager
from src.models.rag import get_rag_chain
from src.config.settings import settings

logger = logging.getLogger("NexusAI.App")

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="Nexus | GenAI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# Custom CSS (Futuristic Dark UI)
# ==========================================
st.markdown("""
<style>
    /* Typography */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Background and Layout */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(11, 15, 25, 1) 0%, rgba(22, 17, 34, 1) 90%);
        color: #e2e8f0;
    }
    
    /* Header Animation & Styling */
    .nexus-header {
        text-align: center;
        padding: 2.5rem 0;
        margin-bottom: 2rem;
    }
    
    .nexus-title {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00f2fe, #4facfe, #9b51e0, #00f2fe);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 4s linear infinite;
        margin-bottom: 0.2rem;
    }
    
    .nexus-subtitle {
        font-size: 1.2rem;
        color: #8b9bb4;
        letter-spacing: 3px;
        text-transform: uppercase;
    }
    
    @keyframes shine {
        to {
            background-position: 200% center;
        }
    }

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.4) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(0, 242, 254, 0.1);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, rgba(79, 172, 254, 0.8) 0%, rgba(0, 242, 254, 0.8) 100%);
        color: white !important;
        border: 1px solid rgba(0, 242, 254, 0.5);
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.2);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 25px rgba(0, 242, 254, 0.5);
        background: linear-gradient(90deg, rgba(79, 172, 254, 1) 0%, rgba(0, 242, 254, 1) 100%);
    }

    /* Chat Messages Glassmorphism */
    .stChatMessage {
        background: rgba(30, 41, 59, 0.3) !important;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    
    [data-testid="chat-message-user"] {
        border-left: 4px solid #9b51e0 !important;
        background: linear-gradient(90deg, rgba(155, 81, 224, 0.05) 0%, transparent 100%) !important;
    }
    
    [data-testid="chat-message-assistant"] {
        border-left: 4px solid #00f2fe !important;
        background: linear-gradient(90deg, rgba(0, 242, 254, 0.05) 0%, transparent 100%) !important;
    }

    /* Chat Input */
    .stChatInput > div {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(0, 242, 254, 0.3) !important;
        border-radius: 24px !important;
        backdrop-filter: blur(15px);
        padding: 0.2rem 1rem;
    }
    .stChatInput > div:focus-within {
        border-color: #00f2fe !important;
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.3) !important;
    }

    /* Source Box */
    .source-box {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(0, 242, 254, 0.2);
        border-radius: 8px;
        padding: 0.8rem;
        font-size: 0.85rem;
        margin-top: 0.5rem;
        color: #a0aec0;
        transition: all 0.3s ease;
    }
    .source-box:hover {
        border-color: rgba(0, 242, 254, 0.5);
        box-shadow: 0 0 10px rgba(0, 242, 254, 0.1);
    }
    .source-box span {
        color: #00f2fe;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Hide elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# Session State
# ==========================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "knowledge_base_ready" not in st.session_state:
    st.session_state.knowledge_base_ready = False

# ==========================================
# Sidebar Configuration
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #00f2fe; margin-bottom: 20px;'>⚙️ Dashboard</h2>", unsafe_allow_html=True)
    
    st.markdown("### 🗄️ Knowledge Base")
    st.caption(f"Syncing documents from `./{settings.DOCS_DIRECTORY}`")
    
    uploaded_files = st.file_uploader(
        "Upload Documents (TXT/PDF)", 
        type=["txt", "pdf"], 
        accept_multiple_files=True
    )
    
    if st.button("Sync & Initialize Vector Store", use_container_width=True):
        with st.spinner("Processing documents & updating ChromaDB..."):
            try:
                # Save newly uploaded files to the docs/ directory
                if uploaded_files:
                    os.makedirs(settings.DOCS_DIRECTORY, exist_ok=True)
                    for file in uploaded_files:
                        file_path = os.path.join(settings.DOCS_DIRECTORY, file.name)
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                        logger.info(f"Saved uploaded file to {file_path}")
                        
                chunks = process_and_get_chunks()
                if not chunks:
                    st.warning("No documents found to index.")
                else:
                    db_manager.add_documents(chunks)
                    st.session_state.knowledge_base_ready = True
                    st.success(f"✅ Indexed {len(chunks)} chunks!")
                    logger.info("Knowledge base initialization successful.")
            except Exception as e:
                logger.error(f"Failed to initialize KB: {e}")
                st.error(f"Initialization Failed: {e}")
                
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 💬 Session Control")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        logger.info("Chat history cleared by user.")
        st.rerun()
        
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style='text-align: center; color: #64748b; font-size: 0.8rem;'>
            <b>LLM:</b> {settings.LLM_MODEL_NAME}<br>
            <b>Vector Store:</b> ChromaDB<br>
            v1.0.0
        </div>
    """, unsafe_allow_html=True)

# ==========================================
# Main Content
# ==========================================
st.markdown("""
<div class="nexus-header">
    <div class="nexus-title">NEXUS AI</div>
    <div class="nexus-subtitle">Advanced Multi-Agent Retrieval System</div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.knowledge_base_ready:
    st.markdown("""
    <div style='text-align: center; margin-top: 40px; padding: 40px; background: rgba(30,41,59,0.3); border-radius: 20px; border: 1px dashed rgba(0, 242, 254, 0.3); backdrop-filter: blur(10px);'>
        <h3 style='color: #e2e8f0; margin-bottom: 15px;'>System Standby</h3>
        <p style='color: #94a3b8; font-size: 1.1em;'>Please initialize the vector store from the sidebar dashboard to load your document contexts and activate the intelligence core.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Display Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Render Sources
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("🔍 View Context Sources"):
                    for idx, src in enumerate(message["sources"], 1):
                        file_name = os.path.basename(src['source'])
                        st.markdown(f"""
                        <div class="source-box">
                            <span>[{idx}] {file_name}</span><br><br>
                            <i>{src['content']}</i>
                        </div>
                        """, unsafe_allow_html=True)

    # Input Field
    if prompt := st.chat_input("Query the intelligence core..."):
        # Append User Message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant Response (Streaming)
        with st.chat_message("assistant"):
            try:
                rag = get_rag_chain()
                stream_generator = rag.stream_answer(prompt, st.session_state.chat_history)
                
                # We need to separate text chunks from source chunks coming from generator
                def text_stream():
                    for chunk in stream_generator:
                        if "text" in chunk:
                            yield chunk["text"]
                        elif "sources" in chunk:
                            # Stash sources in session state temporarily to display after stream
                            st.session_state.current_sources = chunk["sources"]
                
                # Stream the text
                full_response = st.write_stream(text_stream())
                
                # Display Sources if any
                sources = st.session_state.get("current_sources", [])
                if sources:
                    with st.expander("🔍 View Context Sources"):
                        for idx, src in enumerate(sources, 1):
                            file_name = os.path.basename(src['source'])
                            st.markdown(f"""
                            <div class="source-box">
                                <span>[{idx}] {file_name}</span><br><br>
                                <i>{src['content']}</i>
                            </div>
                            """, unsafe_allow_html=True)
                            
                # Save to history
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": full_response,
                    "sources": sources
                })
                # Clean temp state
                if "current_sources" in st.session_state:
                    del st.session_state.current_sources
                    
            except Exception as e:
                logger.error(f"Error during streaming response: {e}")
                st.error(f"System Error: {str(e)}")

# Footer
st.markdown("""
<div style='text-align: center; padding-top: 50px; color: #475569; font-size: 0.8rem;'>
    Nexus AI Platform &copy; 2026 | Powered by Groq & LangChain
</div>
""", unsafe_allow_html=True)
change the code
