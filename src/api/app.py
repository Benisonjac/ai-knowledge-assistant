"""FastAPI application – production-grade REST API for the RAG system."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage

from src.api.schemas import (
    HealthResponse,
    IngestDirectoryRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    StoreStatsResponse,
)
from src.config import logger
from src.ingestion import ingest_directory, ingest_file, get_store_stats
from src.retrieval import query_knowledge_base

# ── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Knowledge Assistant",
    description=(
        "Production-grade RAG API with LangGraph agentic retrieval, "
        "cross-encoder reranking, hallucination checking, and LangSmith observability."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health Check ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse()


# ── Query Endpoint ───────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(request: QueryRequest):
    """Ask a question and get an AI-generated answer grounded in your documents."""
    try:
        # Convert chat history dicts → LangChain message objects
        messages = []
        for msg in request.chat_history:
            if msg["role"] == "human":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

        result = await query_knowledge_base(
            question=request.question,
            chat_history=messages if messages else None,
        )
        return QueryResponse(**result)

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Ingestion Endpoints ─────────────────────────────────────────────────────

@app.post("/ingest/directory", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_dir(request: IngestDirectoryRequest):
    """Ingest all supported documents from a directory."""
    directory = Path(request.directory)
    if not directory.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {request.directory}")

    result = ingest_directory(directory)
    return IngestResponse(**result)


@app.post("/ingest/upload", response_model=IngestResponse, tags=["Ingestion"])
async def upload_and_ingest(file: UploadFile = File(...)):
    """Upload a single document and ingest it into the knowledge base."""
    supported = {".pdf", ".txt", ".md", ".docx", ".pptx"}
    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Supported: {supported}",
        )

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        result = ingest_file(tmp_path)
        return IngestResponse(**result)
    finally:
        tmp_path.unlink(missing_ok=True)


# ── Store Stats ──────────────────────────────────────────────────────────────

@app.get("/store/stats", response_model=StoreStatsResponse, tags=["System"])
async def store_stats():
    """Get statistics about the vector store."""
    return StoreStatsResponse(**get_store_stats())
