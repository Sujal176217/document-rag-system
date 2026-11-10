# Document RAG System

A simple Retrieval-Augmented Generation (RAG) service in Python.
Upload documents, ask questions, get short answers grounded in those documents, with sources.

**Example**
```
Context : Interns get 1 casual leave per month.
Question: How many leaves does an intern get?
Answer  : Interns are entitled to 1 casual leave per month.
```

## Tech stack
| Part | Technology | Why |
|---|---|---|
| Language | Python | Strong RAG ecosystem |
| Framework | LangChain | Connects loaders, retriever and LLM |
| Embeddings | OpenAI API (`text-embedding-3-small`) | No local GPU needed |
| Vector DB | ChromaDB | Lightweight, easy to develop with |
| LLM | Claude (Anthropic API) | No local GPU needed |
| API | FastAPI | Exposes RAG as a REST API |
| Testing | Pytest | Tests run offline with fake models |

## How it works
```
INGEST:  file -> load (loader.py) -> split into chunks (rag.py) -> embed -> ChromaDB
ASK:     question -> embed -> top-k similar chunks -> prompt + LLM -> answer + sources
```

## Project structure
```
app/
  config.py       settings (chunk size, top_k, model names)
  providers.py    embeddings + LLM (the only place that talks to external AI APIs)
  loader.py       .txt / .md / .pdf -> LangChain Documents
  vectorstore.py  ChromaDB setup
  rag.py          core logic: ingest() and ask()
  service.py      builds the real service
  main.py         FastAPI endpoints
  cli.py          command line helper
data/docs/        sample documents (fake data)
tests/            pytest tests (no API keys required)
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then add your API keys
```
Add `chroma_db/` to `.gitignore` so the local database isn't committed.

## Run
```bash
python -m app.cli ingest                              # index data/docs
python -m app.cli ask "How many leaves does an intern get?"

uvicorn app.main:app --reload                         # start the API
# open http://127.0.0.1:8000/docs for interactive Swagger UI
```

### API
| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/ingest` | Upload `.txt/.md/.pdf` files (multipart `files`) |
| POST | `/ask` | `{"question": "..."}` -> `{"answer": "...", "sources": [...]}` |

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many leaves does an intern get?"}'
```

## Test
```bash
pytest
```
Tests use fake embeddings and a fake LLM, so they need no API keys and cost nothing.

## Security notes
- Keep keys only in `.env` (never commit it).
- Text from your documents is sent to the embedding and LLM providers. Use only data you are allowed to share.

## Ideas to extend
Re-ranking, hybrid search, chat history, evaluation set, authentication on the API, Docker.
