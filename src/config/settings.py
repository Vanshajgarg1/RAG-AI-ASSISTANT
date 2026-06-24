import os
import logging
from dotenv import load_dotenv

try:
    import streamlit as st
except Exception:
    st = None

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("NexusAI")


def get_secret(key, default=""):
    # 1. Try Streamlit Secrets
    try:
        if st is not None and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    # 2. Try environment variables
    return os.getenv(key, default)


class Settings:
    GROQ_API_KEY: str = get_secret("GROQ_API_KEY", "")

    LLM_MODEL_NAME: str = get_secret(
        "LLM_MODEL_NAME",
        "llama-3.3-70b-versatile"
    )

    LLM_TEMPERATURE: float = float(
        get_secret("LLM_TEMPERATURE", "0.3")
    )

    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    CHROMA_PERSIST_DIRECTORY: str = "chroma_db"

    DOCS_DIRECTORY: str = "docs"
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 200


settings = Settings()
