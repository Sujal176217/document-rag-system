"""STEP 3/4 - STORE: ChromaDB through LangChain (saved on disk)."""
from langchain_chroma import Chroma


def get_vectorstore(embeddings, persist_dir: str, collection_name: str) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )
