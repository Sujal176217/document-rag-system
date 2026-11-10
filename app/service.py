"""Builds the real RAGService (real APIs + ChromaDB on disk)."""
from .config import settings
from .providers import get_embeddings, get_llm
from .rag import RAGService
from .vectorstore import get_vectorstore


def build_service() -> RAGService:
    store = get_vectorstore(get_embeddings(), settings.chroma_dir, settings.collection_name)
    return RAGService(
        store,
        get_llm(),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
    )
