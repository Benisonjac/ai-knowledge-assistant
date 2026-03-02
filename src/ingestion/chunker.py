"""Advanced text chunking strategies."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings, logger


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Build a recursive character splitter with config-driven parameters."""
    cfg = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller, overlapping chunks for embedding."""
    splitter = get_text_splitter()
    chunks = splitter.split_documents(documents)

    # Add chunk index metadata
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx

    logger.info(
        f"Chunked {len(documents)} documents → {len(chunks)} chunks "
        f"(size={get_settings().chunk_size}, overlap={get_settings().chunk_overlap})"
    )
    return chunks
