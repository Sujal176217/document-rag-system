"""The RAG core: ingest (load -> split -> embed -> store) and ask
(retrieve -> prompt -> LLM). Written with dependency injection so tests can
plug in fake embeddings / a fake LLM and run without any API key."""
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .loader import load_directory, load_file

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided context.\n"
    "- Give a short, direct answer in one or two sentences.\n"
    "- If the context does not contain the answer, reply exactly: "
    "\"I don't know based on the provided documents.\"\n"
    "- Never use outside knowledge."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

NO_DOCS_MESSAGE = "No documents have been indexed yet. Ingest some documents first."


@dataclass
class SourceChunk:
    source: str
    text: str
    page: int | None = None


@dataclass
class Answer:
    answer: str
    sources: list[SourceChunk] = field(default_factory=list)


class RAGService:
    def __init__(self, vectorstore, llm, *, chunk_size=500, chunk_overlap=100, top_k=4):
        self.vectorstore = vectorstore
        self.top_k = top_k
        # STEP 2 - SPLIT text into overlapping chunks
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        # STEP 5 - prompt -> LLM -> plain string (LangChain "LCEL" pipe syntax)
        self.chain = PROMPT | llm | StrOutputParser()

    # ---------------- INGEST ----------------
    def _index(self, docs) -> int:
        chunks = self.splitter.split_documents(docs)
        # Group by source so re-ingesting a file replaces its old chunks.
        by_source: dict[str, list] = {}
        for chunk in chunks:
            by_source.setdefault(chunk.metadata["source"], []).append(chunk)

        for source, source_chunks in by_source.items():
            old = self.vectorstore.get(where={"source": source})
            if old["ids"]:
                self.vectorstore.delete(ids=old["ids"])
            ids = [f"{source}::{i}" for i in range(len(source_chunks))]
            self.vectorstore.add_documents(source_chunks, ids=ids)
        return len(chunks)

    def ingest_file(self, path: Path) -> int:
        return self._index(load_file(path))

    def ingest_directory(self, directory: Path) -> int:
        return self._index(load_directory(directory))

    # ---------------- ASK ----------------
    def retrieve(self, question: str):
        return self.vectorstore.similarity_search(question, k=self.top_k)

    def ask(self, question: str) -> Answer:
        docs = self.retrieve(question)
        if not docs:
            return Answer(NO_DOCS_MESSAGE)

        context = "\n\n".join(
            f"[{i}] (source: {d.metadata.get('source', '?')})\n{d.page_content}"
            for i, d in enumerate(docs, start=1)
        )
        text = self.chain.invoke({"context": context, "question": question})
        sources = [
            SourceChunk(d.metadata.get("source", "?"), d.page_content, d.metadata.get("page"))
            for d in docs
        ]
        return Answer(text.strip(), sources)
