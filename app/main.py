"""FastAPI layer: exposes the RAG service as a REST API.

Run:  uvicorn app.main:app --reload     then open http://127.0.0.1:8000/docs
"""
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .loader import SUPPORTED_EXTENSIONS
from .rag import RAGService
from .service import build_service

app = FastAPI(title="Document RAG System", version="0.1.0")


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


@app.get("/health")
def health():
    return {"status": "ok"}


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
            results[name] = service.ingest_file(path)
    return {"chunks_indexed": results}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, service: RAGService = Depends(get_service)):
    result = service.ask(req.question)
    return AskResponse(
        answer=result.answer,
        sources=[SourceOut(source=s.source, page=s.page, text=s.text) for s in result.sources],
    )
