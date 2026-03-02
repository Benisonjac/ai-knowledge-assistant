"""Reranker – uses FlashRank (lightweight, local cross-encoder reranking)."""

from __future__ import annotations

from functools import lru_cache

from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document

from src.config import get_settings, logger


@lru_cache
def _get_ranker() -> Ranker:
    """Load FlashRank model (tiny, runs on CPU)."""
    return Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./data/flashrank_cache")


def rerank_documents(query: str, documents: list[Document]) -> list[Document]:
    """Rerank retrieved documents using a cross-encoder for better relevance.

    This is a KEY resume skill: demonstrates understanding of two-stage retrieval
    (fast vector search → precise cross-encoder reranking).
    """
    if not documents:
        return []

    cfg = get_settings()
    ranker = _get_ranker()

    # Convert LangChain docs → FlashRank format
    passages = [
        {"id": str(i), "text": doc.page_content, "meta": doc.metadata}
        for i, doc in enumerate(documents)
    ]

    rerank_request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_request)

    # Map back to LangChain Documents, sorted by relevance
    reranked: list[Document] = []
    for result in results[: cfg.rerank_top_n]:
        idx = int(result["id"])
        doc = documents[idx]
        doc.metadata["rerank_score"] = float(result["score"])
        reranked.append(doc)

    logger.info(
        f"Reranked {len(documents)} → top {len(reranked)} documents "
        f"(scores: {[round(d.metadata['rerank_score'], 3) for d in reranked]})"
    )
    return reranked
