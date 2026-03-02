"""Retriever – vector similarity search + reranking."""

from __future__ import annotations

from langchain_core.documents import Document

from src.config import get_settings, logger
from src.ingestion.vector_store import get_vector_store
from src.retrieval.reranker import rerank_documents


def retrieve(query: str, use_reranker: bool = True) -> list[Document]:
    """Two-stage retrieval: vector search → cross-encoder reranking.

    Stage 1: Retrieve top-K candidates via cosine similarity (fast, approximate).
    Stage 2: Rerank candidates with a cross-encoder model (precise, slower).
    """
    cfg = get_settings()
    store = get_vector_store()

    # Stage 1 – Dense retrieval
    logger.info(f"Retrieving top-{cfg.top_k_results} candidates for: '{query[:80]}...'")
    candidates = store.similarity_search_with_relevance_scores(query, k=cfg.top_k_results)

    if not candidates:
        logger.warning("No documents found in vector store")
        return []

    # Attach similarity score to metadata
    docs: list[Document] = []
    for doc, score in candidates:
        doc.metadata["similarity_score"] = round(float(score), 4)
        docs.append(doc)

    logger.info(f"Stage 1 returned {len(docs)} candidates")

    # Stage 2 – Reranking
    if use_reranker and len(docs) > 1:
        docs = rerank_documents(query, docs)

    return docs


def format_context(documents: list[Document]) -> str:
    """Format retrieved documents into a context string for the LLM."""
    context_parts: list[str] = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source_file", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        context_parts.append(
            f"[Source {i}: {source} | Chunk {chunk_idx}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(context_parts)
