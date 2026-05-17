import logging
from typing import List
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from src.config.settings import settings

logger = logging.getLogger("NexusAI.VectorStore")

class ChromaDBManager:
    """
    Manages the ChromaDB vector storage, including embedding initialization,
    document ingestion, and retrieval.
    """
    def __init__(self):
        self.persist_directory = settings.CHROMA_PERSIST_DIRECTORY
        logger.info(f"Initializing embedding model: {settings.EMBEDDING_MODEL}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None
        self._initialize_store()

    def _initialize_store(self) -> None:
        """Connects to the persistent Chroma vector store."""
        try:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            logger.info("ChromaDB vector store connected successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise e
        
    def add_documents(self, documents: List[Document]) -> None:
        """
        Embeds and adds document chunks to the ChromaDB, then persists them.
        
        Args:
            documents (List[Document]): The chunked documents to ingest.
        """
        if not documents:
            logger.warning("No documents provided to add_documents.")
            return
            
        try:
            logger.info(f"Adding {len(documents)} chunks to ChromaDB...")
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            logger.info("Documents successfully persisted to ChromaDB.")
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            raise e
        
    def get_retriever(self, k: int = 5):
        """
        Returns a retriever interface for RAG operations using MMR for diversity.
        
        Args:
            k (int): Number of top documents to retrieve.
            
        Returns:
            VectorStoreRetriever: The LangChain retriever object.
        """
        return self.vector_store.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": k}
        )

# Create a singleton instance to be imported across the app
db_manager = ChromaDBManager()
