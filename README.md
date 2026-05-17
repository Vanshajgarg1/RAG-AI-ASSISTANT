# Nexus AI — RAG AI Assistant

A production-ready Retrieval-Augmented Generation (RAG) AI Assistant built using Python, Streamlit, LangChain, ChromaDB, and GROQ LLMs.

Nexus AI allows users to query custom documents using natural language by combining semantic search with large language models.

---

# 🚀 Features

- 📄 Document ingestion pipeline
- 🧠 Retrieval-Augmented Generation (RAG)
- 🔍 Semantic similarity search
- 🗂 Persistent ChromaDB vector storage
- ⚡ GROQ ultra-fast LLM inference
- 💬 Conversational AI chat interface
- 🌌 Futuristic dark UI
- 🧾 Source-aware responses
- 🧩 Modular project architecture
- 📚 Automatic document chunking
- 🔄 Real-time response streaming
- 🎨 Glassmorphism Streamlit design

---

# 🏗 Architecture

The system follows a modular RAG pipeline architecture:

1. Documents are loaded from the `/docs` directory
2. Text is split into chunks
3. Embeddings are generated using HuggingFace models
4. Vectors are stored in ChromaDB
5. User queries are embedded
6. Retriever finds relevant context
7. Context is passed to GROQ LLM
8. AI generates grounded responses

---

# 📂 Project Structure

```bash
.
├── app.py
├── requirements.txt
├── README.md
├── .env
├── docs/
├── chroma_db/
└── src/
    ├── config/
    │   └── settings.py
    ├── database/
    │   └── vector_store.py
    ├── models/
    │   ├── llm.py
    │   └── rag.py
    └── utils/
        └── document_loader.py
```

---

# 🛠 Tech Stack

| Technology | Purpose |
|------------|----------|
| Python | Core Programming Language |
| Streamlit | Frontend UI |
| LangChain | RAG Framework |
| ChromaDB | Vector Database |
| GROQ API | LLM Inference |
| HuggingFace Embeddings | Semantic Embeddings |
| RecursiveCharacterTextSplitter | Text Chunking |
| dotenv | Environment Variables |

---

# 🧠 LLM Configuration

### Model Used

```python
llama3-70b-8192
```

### Embedding Model

```python
sentence-transformers/all-MiniLM-L6-v2
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/nexus-ai.git

cd nexus-ai
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key
```

---

# 📄 Add Documents

Place your `.txt` documents inside:

```bash
/docs
```

Example:

```bash
docs/
├── ibm-multi-agent.txt
└── readytensor-multi-agent.txt
```

---

# ▶️ Run Application

```bash
streamlit run app.py
```

---

# 🌐 Streamlit UI Features

- Dark futuristic interface
- Glassmorphism components
- Sidebar controls
- Chat history
- Animated loading indicators
- AI typing effect
- Source references
- Responsive layout

---

# 🔄 RAG Pipeline Workflow

```text
User Question
      ↓
Query Embedding
      ↓
ChromaDB Similarity Search
      ↓
Relevant Context Retrieval
      ↓
Prompt Construction
      ↓
GROQ LLM Response
      ↓
Grounded AI Answer
```

---

# 📦 Requirements

```txt
streamlit
langchain
langchain-community
langchain-core
langchain-groq
chromadb
sentence-transformers
python-dotenv
pypdf
```

---

# 🧩 Core Modules

## document_loader.py

Responsible for:
- Loading documents
- Splitting text
- Creating chunks
- Managing metadata

---

## vector_store.py

Responsible for:
- ChromaDB initialization
- Vector storage
- Similarity search
- Persistent embeddings

---

## llm.py

Responsible for:
- GROQ API integration
- LLM configuration
- Prompt handling
- Streaming responses

---

## rag.py

Responsible for:
- Retrieval chain
- Context injection
- Question answering pipeline
- Response formatting

---

# 🎯 Future Improvements

- PDF upload support
- Multi-document chat
- Conversation memory
- Hybrid search
- Agentic workflows
- Web search fallback
- Authentication system
- Docker deployment
- Kubernetes support

---

# ☁️ Deployment

## Streamlit Cloud

1. Push project to GitHub
2. Open Streamlit Cloud
3. Connect repository
4. Add environment variables
5. Deploy app

---

# 🖼 Example Use Cases

- AI research assistant
- Company knowledge base
- Technical documentation chatbot
- PDF question-answering system
- Internal enterprise AI assistant
- Educational AI tutor

---

# 📸 UI Preview

- Futuristic AI dashboard
- Neon-glow interface
- Modern chat layout
- Intelligent document retrieval

---

# 👨‍💻 Developer

Built with ❤️ using:
- LangChain
- GROQ
- ChromaDB
- Streamlit

---

# 📜 License

MIT License

---

# ⭐ Support

If you like this project:

- Star the repository
- Fork the project
- Share on LinkedIn
- Contribute improvements

---

# 🔥 Nexus AI

> "Connecting Intelligence Through Retrieval-Augmented Generation"
