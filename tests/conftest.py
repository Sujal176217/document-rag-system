"""Test fixtures. No API keys needed: we use fake embeddings and a fake LLM."""
import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.runnables import RunnableLambda

from app.rag import RAGService


@pytest.fixture
def echo_llm():
    """A fake 'LLM' that returns the full prompt it received.
    That lets tests check exactly what context the model would have seen."""
    return RunnableLambda(lambda prompt_value: prompt_value.to_string())


@pytest.fixture
def service(tmp_path, echo_llm):
    store = Chroma(
        collection_name="test",
        embedding_function=DeterministicFakeEmbedding(size=64),
        persist_directory=str(tmp_path / "chroma"),
    )
    return RAGService(store, echo_llm, chunk_size=200, chunk_overlap=20, top_k=3)


@pytest.fixture
def policy_file(tmp_path):
    path = tmp_path / "policy.txt"
    path.write_text("Interns get 1 casual leave per month.", encoding="utf-8")
    return path
