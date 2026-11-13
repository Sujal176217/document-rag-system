"""FastAPI layer: web UI at /, REST API, Swagger docs at /docs.

Run:  uvicorn app.main:app --reload     then open http://127.0.0.1:8000
"""
import logging
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .loader import SUPPORTED_EXTENSIONS
from .providers import ConfigError, describe_providers
from .rag import RAGService
from .service import build_service

logger = logging.getLogger("uvicorn.error")
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Document RAG System", version="0.2.0")


@lru_cache
def get_service() -> RAGService:
    return build_service()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class SourceOut(BaseModel):
    source: str
    page: int | None = None
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/providers")
def providers():
    """Active AI providers and which keys are set (never shows keys)."""
    return describe_providers()


@app.exception_handler(ConfigError)
def config_error_handler(request, exc: ConfigError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


def _failure(action: str, exc: Exception) -> HTTPException:
    logger.exception("%s failed", action)
    return HTTPException(
        502,
        f"{action} failed ({type(exc).__name__}). Check your API key, quota or "
        "network, and see the server log for details.",
    )


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...), service: RAGService = Depends(get_service)):
    """Upload one or more .txt / .md / .pdf files and index them."""
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            name = Path(f.filename or "").name  # strip any path for safety
            if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise HTTPException(400, f"Unsupported file type: {name}")
            path = Path(tmp) / name
            path.write_bytes(await f.read())
            try:
                results[name] = service.ingest_file(path)
            except ConfigError:
                raise
            except Exception as exc:
                raise _failure(f"Indexing {name}", exc) from exc
    return {"chunks_indexed": results}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, service: RAGService = Depends(get_service)):
    try:
        result = service.ask(req.question)
    except ConfigError:
        raise
    except Exception as exc:
        raise _failure("Answering", exc) from exc
    return AskResponse(
        answer=result.answer,
        sources=[SourceOut(source=s.source, page=s.page, text=s.text) for s in result.sources],
    )
