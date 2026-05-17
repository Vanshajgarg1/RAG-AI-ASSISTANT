import logging
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from src.models.llm import get_llm
from src.database.vector_store import db_manager

logger = logging.getLogger("NexusAI.RAG")

SYSTEM_PROMPT = """
You are an intelligent coding and document assistant.

Rules:
- Answer ONLY from the provided context.
- If the context is insufficient, say:
  'The uploaded documents do not contain enough information.'
- Explain coding concepts clearly.
- Provide examples if available in context.
- Combine information from multiple chunks if needed.
- Keep responses concise and professional.

Context:
{context}

Question:
{question}

Answer:
"""

class RAGChain:
    """
    Encapsulates the conversational RAG logic with deduplication and custom prompt.
    """
    def __init__(self):
        logger.info("Initializing RAG Chain...")
        self.llm = get_llm(streaming=True)
        self.retriever = db_manager.get_retriever(k=5)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        logger.info("RAG Chain initialized successfully.")
        
    def _deduplicate_docs(self, docs: List[Document]) -> List[Document]:
        """Removes documents with identical page_content."""
        seen = set()
        unique_docs = []
        for doc in docs:
            content_hash = hash(doc.page_content.strip())
            if content_hash not in seen:
                seen.add(content_hash)
                unique_docs.append(doc)
        return unique_docs

    def stream_answer(self, question: str, chat_history: List[Dict[str, str]]):
        """
        Streams the answer back to the UI chunk by chunk.
        Yields context source metadata first, then text chunks.
        """
        formatted_history = self._format_history(chat_history)
        logger.info(f"Invoking retrieval chain for query: '{question}'")
        
        # 1. Retrieve and deduplicate
        raw_docs = self.retriever.invoke(question)
        docs = self._deduplicate_docs(raw_docs)
        logger.info(f"Retrieved {len(raw_docs)} chunks, reduced to {len(docs)} unique chunks.")
        
        # 2. Extract sources and yield them
        sources = []
        context_parts = []
        for doc in docs:
            source_name = doc.metadata.get("source", "Unknown Source")
            page_num = doc.metadata.get("page", "N/A")
            snippet = doc.page_content[:150] + "..."
            sources.append({
                "source": source_name, 
                "page": page_num,
                "content": snippet
            })
            context_parts.append(doc.page_content)
            
        yield {"sources": sources}
        
        # 3. Format context and invoke LLM
        context_str = "\n\n".join(context_parts)
        chain = self.prompt | self.llm
        
        # 4. Stream response
        for chunk in chain.stream({
            "context": context_str,
            "question": question,
            "chat_history": formatted_history
        }):
            if chunk.content:
                yield {"text": chunk.content}

    def _format_history(self, history: List[Dict[str, str]]) -> List:
        """Converts raw dictionary history to LangChain message formats."""
        messages = []
        for msg in history[-10:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        return messages

# Singleton instance
rag_chain = None

def get_rag_chain():
    """Lazy loader for the RAG chain."""
    global rag_chain
    if rag_chain is None:
        rag_chain = RAGChain()
    return rag_chain
