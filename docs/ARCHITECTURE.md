# Architecture

```
                       +---------------------------+
   REST client  --->   |  app/main.py  (FastAPI)   |   /ingest  /ask  /providers  /health
   CLI          --->   |  app/cli.py               |
                       +-------------+-------------+
                                     |
                              +------v------+
                              | app/rag.py  |   RAGService: ingest() + ask()
                              +--+-------+--+   (knows nothing about WHICH provider)
                                 |       |
              +------------------+       +---------------------+
              |                                                |
     +--------v---------+                          +-----------v-----------+
     | vectorstore.py   |                          |  providers/ package   |
     | ChromaDB on disk |                          |  registry + factory   |
     +------------------+                          +-----------+-----------+
                                                               |
                                   +---------------------------+-------------------+
                                   |                           |                   |
                              Google Gemini                 OpenAI             Anthropic
                         (embeddings + LLM)          (embeddings + LLM)         (LLM only)
```

## Design rules
1. **Core logic is provider-agnostic.** `rag.py` receives a vector store and an LLM
   as arguments (dependency injection). It never imports Google/OpenAI/Anthropic.
2. **One registry.** `providers/registry.py` lists providers; `providers/builtin.py`
   fills it. Config picks providers by *name*.
3. **Secrets live only in `.env`.** `/providers` and `python -m app.cli providers`
   show which keys are set, never their values.
4. **One vector collection per embedding model.** The collection name contains the
   provider+model (e.g. `document_rag_google-gemini-embedding-001`). Vectors from
   different models are different "languages" and must never be mixed.
5. **Everything is testable offline.** Tests inject fake embeddings, a fake LLM and
   fake providers; no key or network needed.

## Multi-provider features
| Need | How |
|---|---|
| Switch provider | change `LLM_PROVIDER` / `EMBEDDING_PROVIDER` in `.env` |
| Several keys for one provider | `GOOGLE_API_KEY=key1,key2` - if key1 fails (quota/rate limit), key2 is tried |
| Backup provider | `LLM_FALLBACKS=anthropic,openai` - used only if the main LLM call fails |
| Add a new provider | one `register_provider(ProviderSpec(...))` call (see below) |

Embeddings have **no automatic fallback** between *different* models, because their
vectors are incompatible. Switching embedding model = new collection = re-run ingest.

## Add a new provider (example: Cohere)
1. `pip install langchain-cohere` and add it to `requirements.txt`.
2. In `app/providers/builtin.py` add:
```python
def _cohere_llm(model, api_key):
    from langchain_cohere import ChatCohere
    return ChatCohere(model=model, cohere_api_key=api_key)

register_provider(ProviderSpec(
    name="cohere", key_env="COHERE_API_KEY",
    llm_factory=_cohere_llm, default_llm_model="<model-name>",
))
```
3. Put `COHERE_API_KEY=...` in `.env` and set `LLM_PROVIDER=cohere`.
4. Add a line to `tests/test_providers.py` if you like. Nothing else changes.

## Future ideas
- Per-request provider choice (`POST /ask {"provider": "openai"}`)
- Authentication + rate limiting on the API
- Per-user / per-team collections (multi-tenant)
- Re-ranking, hybrid search, evaluation set, Docker
