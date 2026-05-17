import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env, overriding existing ones
load_dotenv(override=True)

# Configure Application Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NexusAI")

class Settings:
    """
    Configuration settings for the RAG AI Assistant.
    Loads variables from the environment with sensible defaults.
    """
    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # LLM Settings
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "llama-3.3-70b-versatile")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    
    # Embedding Settings
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    
    # Database Settings
    CHROMA_PERSIST_DIRECTORY: str = "chroma_db"
    
    # Data Settings
    DOCS_DIRECTORY: str = "docs"
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 200

settings = Settings()
