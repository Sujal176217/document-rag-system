"""All settings in one place (read from environment / .env).

Providers are chosen by NAME (google, openai, anthropic, ...). API keys are
NOT stored here - each provider reads its own key variable (see providers/).
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]


def _csv(name: str) -> tuple[str, ...]:
    return tuple(p.strip().lower() for p in os.getenv(name, "").split(",") if p.strip())


@dataclass(frozen=True)
class Settings:
    # --- which providers to use (empty model = that provider's default) ---
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "google").strip().lower()
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "").strip()
    llm_provider: str = os.getenv("LLM_PROVIDER", "google").strip().lower()
    llm_model: str = os.getenv("LLM_MODEL", "").strip()
    # Backup LLM providers, tried in order if the primary call fails.
    llm_fallbacks: tuple[str, ...] = field(default_factory=lambda: _csv("LLM_FALLBACKS"))

    # --- storage ---
    chroma_dir: str = os.getenv("CHROMA_DIR", str(ROOT / "chroma_db"))
    collection_prefix: str = "document_rag"
    docs_dir: Path = ROOT / "data" / "docs"

    # --- RAG tuning ---
    chunk_size: int = 500      # characters per chunk
    chunk_overlap: int = 100   # shared characters between neighbouring chunks
    top_k: int = 4             # chunks handed to the LLM


settings = Settings()
