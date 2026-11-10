import pytest

from app.loader import load_file
from app.rag import NO_DOCS_MESSAGE


def test_loader_reads_text(policy_file):
    docs = load_file(policy_file)
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "policy.txt"


def test_loader_rejects_unknown_type(tmp_path):
    bad = tmp_path / "data.exe"
    bad.write_text("x")
    with pytest.raises(ValueError):
        load_file(bad)


def test_ask_before_ingest_returns_friendly_message(service):
    assert service.ask("anything?").answer == NO_DOCS_MESSAGE


def test_ingest_counts_chunks(service, policy_file):
    assert service.ingest_file(policy_file) == 1


def test_context_and_question_reach_the_llm(service, policy_file):
    service.ingest_file(policy_file)
    result = service.ask("How many leaves does an intern get?")
    # The echo LLM returns the prompt, so we can inspect what it was given.
    assert "Interns get 1 casual leave per month." in result.answer
    assert "How many leaves does an intern get?" in result.answer
    assert result.sources[0].source == "policy.txt"


def test_reingest_does_not_duplicate(service, policy_file):
    service.ingest_file(policy_file)
    service.ingest_file(policy_file)
    assert len(service.vectorstore.get()["ids"]) == 1


def test_long_document_is_split(service, tmp_path):
    long_file = tmp_path / "long.txt"
    long_file.write_text("Sentence about policy. " * 100, encoding="utf-8")
    assert service.ingest_file(long_file) > 1
