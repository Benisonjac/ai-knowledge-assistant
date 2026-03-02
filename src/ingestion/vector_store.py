"""ChromaDB vector store – persistent, local vector database."""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import get_settings, logger
from src.ingestion.embeddings import get_embedding_model


def get_vector_store() -> Chroma:
    """Return a Chroma vector store backed by local persistence."""
    cfg = get_settings()
    cfg.chroma_path.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=cfg.chroma_collection_name,
        embedding_function=get_embedding_model(),
        persist_directory=str(cfg.chroma_path),
    )


def add_documents_to_store(documents: list[Document]) -> int:
    """Embed and insert documents into ChromaDB. Returns count added."""
    store = get_vector_store()

    # Deduplicate by page_content hash to avoid re-indexing
    existing_ids = set(store.get()["ids"]) if store.get()["ids"] else set()
    new_docs = []
    new_ids = []

    for doc in documents:
        doc_id = str(hash(doc.page_content))
        if doc_id not in existing_ids:
            new_docs.append(doc)
            new_ids.append(doc_id)

    if new_docs:
        store.add_documents(documents=new_docs, ids=new_ids)
        logger.info(f"Added {len(new_docs)} new chunks to vector store")
    else:
        logger.info("No new documents to add – all chunks already indexed")

    return len(new_docs)


def get_store_stats() -> dict:
    """Return basic statistics about the vector store."""
    store = get_vector_store()
    collection = store.get()
    return {
        "total_chunks": len(collection["ids"]),
        "collection_name": get_settings().chroma_collection_name,
    }
