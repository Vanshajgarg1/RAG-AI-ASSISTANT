import os
import logging
from glob import glob
from typing import List
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config.settings import settings

logger = logging.getLogger("NexusAI.DocumentLoader")

def load_documents() -> List[Document]:
    """
    Loads all .txt and .pdf files from the designated docs directory.
    
    Returns:
        List[Document]: A list of LangChain Document objects containing text and metadata.
    """
    documents: List[Document] = []
    
    # Load TXT files
    txt_paths = glob(os.path.join(settings.DOCS_DIRECTORY, "*.txt"))
    for file_path in txt_paths:
        try:
            logger.info(f"Loading TXT document: {file_path}")
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
        except Exception as e:
            logger.error(f"Error loading TXT {file_path}: {e}")
            
    # Load PDF files
    pdf_paths = glob(os.path.join(settings.DOCS_DIRECTORY, "*.pdf"))
    for file_path in pdf_paths:
        try:
            logger.info(f"Loading PDF document: {file_path}")
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            
    if not documents:
        logger.warning(f"No supported documents found in directory: {settings.DOCS_DIRECTORY}")

    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Splits loaded documents into smaller chunks for optimal vector indexing.
    
    Args:
        documents (List[Document]): The raw documents to split.
        
    Returns:
        List[Document]: The chunked documents.
    """
    if not documents:
        return []
        
    logger.info(f"Splitting {len(documents)} documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks.")
    return chunks

def process_and_get_chunks() -> List[Document]:
    """
    Pipeline that loads and splits documents in one go.
    
    Returns:
        List[Document]: Final document chunks ready for embedding.
    """
    docs = load_documents()
    chunks = split_documents(docs)
    return chunks
