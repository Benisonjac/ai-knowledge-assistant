"""AI Knowledge Assistant – Configuration via pydantic-settings."""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised, type-safe configuration loaded from .env / environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── HuggingFace (free Inference API) ──────────────────────────────
    huggingfacehub_api_token: str = ""

    # ── LangSmith Observability ──────────────────────────────────────────
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "ai-knowledge-assistant"

    # ── ChromaDB ─────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "knowledge_base"

    # ── Embedding ────────────────────────────────────────────────────────
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── LLM (HuggingFace Inference API – free tier) ──────────────────────
    llm_model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048

    # ── Chunking ─────────────────────────────────────────────────────────
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ── Retrieval ────────────────────────────────────────────────────────
    top_k_results: int = 10
    rerank_top_n: int = 4

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_persist_dir)


@lru_cache
def get_settings() -> Settings:
    """Cached singleton so settings are loaded once."""
    return Settings()
