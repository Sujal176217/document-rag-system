"""Builds the real RAGService (real APIs + ChromaDB on disk)."""
import re

from .config import settings
from .providers import embedding_id, get_embeddings, get_llm
from .rag import RAGService
from .vectorstore import get_vectorstore


def collection_name() -> str:
    # One collection per embedding model: switching model never mixes vectors
    # of different sizes/meanings. (Re-run ingest after switching.)
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", f"{settings.collection_prefix}_{embedding_id(settings)}")


def build_service() -> RAGService:
    store = get_vectorstore(get_embeddings(), settings.chroma_dir, collection_name())
    return RAGService(
        store,
        get_llm(),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
    )
