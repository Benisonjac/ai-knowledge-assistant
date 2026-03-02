"""Pydantic schemas for the API layer."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Request Models ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="The user's question")
    chat_history: list[dict[str, str]] = Field(
        default_factory=list,
        description="Previous messages as [{role: 'human'|'ai', content: '...'}]",
    )


class IngestDirectoryRequest(BaseModel):
    directory: str = Field(..., description="Path to directory containing documents")


# ── Response Models ──────────────────────────────────────────────────────────

class SourceReference(BaseModel):
    file: str
    chunk: int | str
    similarity_score: float | None = None
    rerank_score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    is_grounded: bool
    question: str


class IngestResponse(BaseModel):
    status: str
    documents_loaded: int = 0
    chunks_created: int = 0
    chunks_added: int = 0
    total_chunks_in_store: int = 0


class StoreStatsResponse(BaseModel):
    total_chunks: int
    collection_name: str


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
