import logging
from langchain_groq import ChatGroq
from src.config.settings import settings

logger = logging.getLogger("NexusAI.LLM")

def get_llm(streaming: bool = True) -> ChatGroq:
    """
    Initializes and returns the Groq LLM wrapper instance.
    
    Args:
        streaming (bool): Whether to enable streaming output for the LLM.
        
    Returns:
        ChatGroq: Configured Groq LLM instance.
    """
    if not settings.GROQ_API_KEY:
        error_msg = "GROQ_API_KEY is not set in the environment variables."
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    logger.info(f"Initializing Groq LLM: {settings.LLM_MODEL_NAME} (Streaming={streaming})")
    
    try:
        llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name=settings.LLM_MODEL_NAME,
            temperature=settings.LLM_TEMPERATURE,
            streaming=streaming
        )
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize Groq LLM: {e}")
        raise e
