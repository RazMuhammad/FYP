from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import functools
from typing import Dict, Any, Optional

# Singleton instance
_EMBEDDING_MODEL = None

def set_embeddings():
    """
    Initialize the embedding model as a singleton.
    
    Returns:
        The embedding model instance
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = FastEmbedEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            cache_dir="./embedding_cache"  # Cache embeddings locally
        )
    return _EMBEDDING_MODEL

def get_embedding_model():
    """
    Get the existing embedding model or create a new one.
    
    Returns:
        The embedding model instance
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = set_embeddings()
    return _EMBEDDING_MODEL

def embed_text(text: str) -> list:
    """
    Generate embeddings for a given text.
    
    Args:
        text: The text to embed
        
    Returns:
        List containing the embedding vector
    """
    model = get_embedding_model()
    result = model.embed_query(text)
    return result