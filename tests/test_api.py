import pytest
from fastapi.testclient import TestClient

from app.main import app, get_service


@pytest.fixture
def client(service):
    app.dependency_overrides[get_service] = lambda: service  # use the fake service
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_then_ask(client):
    files = {"files": ("policy.txt", b"Interns get 1 casual leave per month.", "text/plain")}
    r = client.post("/ingest", files=files)
    assert r.status_code == 200
    assert r.json()["chunks_indexed"]["policy.txt"] == 1

    r = client.post("/ask", json={"question": "How many leaves does an intern get?"})
    assert r.status_code == 200
    body = r.json()
    assert "1 casual leave per month" in body["answer"]
    assert body["sources"][0]["source"] == "policy.txt"


def test_rejects_unsupported_file(client):
    files = {"files": ("virus.exe", b"x", "application/octet-stream")}
    assert client.post("/ingest", files=files).status_code == 400


def test_empty_question_is_rejected(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_providers_endpoint_hides_keys(client, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "TOP-SECRET")
    body = client.get("/providers").json()
    assert "TOP-SECRET" not in str(body)
    assert {"google", "openai", "anthropic"} <= {p["name"] for p in body["providers"]}
    assert "active" in body


def test_root_serves_the_web_ui(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Document RAG" in r.text


def test_provider_failure_returns_502_with_message(client, service, monkeypatch):
    def boom(question):
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(service, "ask", boom)
    r = client.post("/ask", json={"question": "hi"})
    assert r.status_code == 502
    assert "RuntimeError" in r.json()["detail"]
