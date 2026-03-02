"""Embedding model factory – uses HuggingFace Inference API (free, no local resources)."""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEndpointEmbeddings

from src.config import get_settings, logger


@lru_cache
def get_embedding_model() -> HuggingFaceEndpointEmbeddings:
    """Return a cached HuggingFace embedding model via the free Inference API."""
    cfg = get_settings()
    logger.info(f"Loading embedding model (remote): {cfg.embedding_model_name}")
    return HuggingFaceEndpointEmbeddings(
        model=cfg.embedding_model_name,
        huggingfacehub_api_token=cfg.huggingfacehub_api_token,
    )
