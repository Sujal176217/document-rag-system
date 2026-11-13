# Document RAG System

A simple, **multi-provider** Retrieval-Augmented Generation (RAG) service in Python.
Upload documents, ask questions, get short answers grounded in those documents, with sources.

```
Context : Interns get 1 casual leave per month.
Question: How many leaves does an intern get?
Answer  : Interns are entitled to 1 casual leave per month.
```

## Tech stack
| Part | Technology | Notes |
|---|---|---|
| Language | Python | |
| Framework | LangChain | loaders, splitter, vector store, chains |
| Embeddings | **Google Gemini** (default), OpenAI | API-based, no local GPU |
| LLM | **Gemini Flash** (default), OpenAI, Claude | API-based, with optional fallbacks |
| Vector DB | ChromaDB | one collection per embedding model |
| API | FastAPI | |
| Tests | Pytest | run offline with fake models |

Providers are **pluggable**: see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then add your key(s)
```
Add `chroma_db/` to `.gitignore`. Get a Google key at https://aistudio.google.com/apikey

## Run
```bash
python -m app.cli providers                           # check what is configured
python -m app.cli ingest                              # index data/docs
python -m app.cli ask "How many leaves does an intern get?"

uvicorn app.main:app --reload                         # web UI at http://127.0.0.1:8000  (Swagger API docs at /docs)
```

### API
| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/providers` | Active providers + which keys are set (never shows keys) |
| POST | `/ingest` | Upload `.txt/.md/.pdf` (multipart `files`) |
| POST | `/ask` | `{"question": "..."}` -> `{"answer": "...", "sources": [...]}` |

## Switching or adding providers
Edit `.env` only:
```
LLM_PROVIDER=google            # google | openai | anthropic
EMBEDDING_PROVIDER=google      # google | openai   (anthropic has no embeddings)
LLM_FALLBACKS=anthropic        # backup if the main LLM call fails
GOOGLE_API_KEY=key1,key2       # several keys: next one is tried on failure
```
**After changing EMBEDDING_PROVIDER or EMBEDDING_MODEL, run ingest again** (new collection).

## Project structure
```
app/
  config.py            settings
  providers/           registry.py (who exists) | builtin.py (Google/OpenAI/Anthropic)
                       factory.py (build embeddings + LLM, keys, fallbacks)
  loader.py            .txt / .md / .pdf -> Documents
  vectorstore.py       ChromaDB
  rag.py               core: ingest() and ask()
  service.py           builds the real service
  main.py              FastAPI endpoints
  cli.py               command line helper
data/docs/             sample documents (fake data)
docs/ARCHITECTURE.md   design + how to add a provider
tests/                 pytest (no keys needed)
```

## Test
```bash
pytest
```

## Security notes
- Keys only in `.env` (git-ignored). Run `git status` before every commit.
- Document text is sent to the chosen providers. Use only data you are allowed to share.
