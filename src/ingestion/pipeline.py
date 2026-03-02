"""Ingestion pipeline – orchestrates loading → chunking → embedding → storing."""

from __future__ import annotations

from pathlib import Path

from src.config import logger
from src.ingestion.document_loader import load_documents_from_directory, load_single_document
from src.ingestion.chunker import chunk_documents
from src.ingestion.vector_store import add_documents_to_store, get_store_stats


def ingest_directory(directory: str | Path) -> dict:
    """Full ingestion pipeline for a directory of documents."""
    directory = Path(directory)
    logger.info(f"Starting ingestion pipeline for: {directory}")

    # 1. Load
    raw_docs = load_documents_from_directory(directory)
    if not raw_docs:
        return {"status": "no_documents", "documents_loaded": 0, "chunks_added": 0}

    # 2. Chunk
    chunks = chunk_documents(raw_docs)

    # 3. Embed & Store
    added = add_documents_to_store(chunks)

    stats = get_store_stats()
    result = {
        "status": "success",
        "documents_loaded": len(raw_docs),
        "chunks_created": len(chunks),
        "chunks_added": added,
        "total_chunks_in_store": stats["total_chunks"],
    }
    logger.info(f"Ingestion complete: {result}")
    return result


def ingest_file(file_path: str | Path) -> dict:
    """Full ingestion pipeline for a single file."""
    file_path = Path(file_path)
    logger.info(f"Ingesting single file: {file_path.name}")

    raw_docs = load_single_document(file_path)
    if not raw_docs:
        return {"status": "unsupported_or_empty", "chunks_added": 0}

    chunks = chunk_documents(raw_docs)
    added = add_documents_to_store(chunks)

    stats = get_store_stats()
    return {
        "status": "success",
        "documents_loaded": len(raw_docs),
        "chunks_created": len(chunks),
        "chunks_added": added,
        "total_chunks_in_store": stats["total_chunks"],
    }
