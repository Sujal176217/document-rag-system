"""STEP 1 - LOAD: turn files into LangChain Document objects."""
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_file(path: Path) -> list[Document]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{suffix}'. Use: {sorted(SUPPORTED_EXTENSIONS)}")

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
        return [
            Document(page_content=text, metadata={"source": path.name, "page": n})
            for n, text in pages
            if text.strip()
        ]

    text = path.read_text(encoding="utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": path.name})] if text.strip() else []


def load_directory(directory: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.extend(load_file(path))
    return docs
