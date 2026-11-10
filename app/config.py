"""All settings in one place (read from environment / .env)."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    llm_model: str = os.getenv("LLM_MODEL", "claude-sonnet-4-6")

    chroma_dir: str = os.getenv("CHROMA_DIR", str(ROOT / "chroma_db"))
    collection_name: str = "document_rag"
    docs_dir: Path = ROOT / "data" / "docs"

    chunk_size: int = 500      # characters per chunk
    chunk_overlap: int = 100   # shared characters between neighbouring chunks
    top_k: int = 4             # chunks handed to the LLM
    max_tokens: int = 500


settings = Settings()
