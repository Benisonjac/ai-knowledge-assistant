"""Tests for the ingestion pipeline."""

from pathlib import Path
import tempfile

import pytest

from src.ingestion.document_loader import load_single_document, SUPPORTED_EXTENSIONS
from src.ingestion.chunker import chunk_documents


class TestDocumentLoader:
    """Tests for document loading."""

    def test_supported_extensions(self):
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".txt" in SUPPORTED_EXTENSIONS
        assert ".md" in SUPPORTED_EXTENSIONS
        assert ".docx" in SUPPORTED_EXTENSIONS

    def test_load_txt_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test document with some content for testing RAG.")
            f.flush()
            docs = load_single_document(Path(f.name))
            assert len(docs) >= 1
            assert "test document" in docs[0].page_content
            assert docs[0].metadata["file_type"] == ".txt"

    def test_unsupported_file(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            docs = load_single_document(Path(f.name))
            assert docs == []


class TestChunker:
    """Tests for document chunking."""

    def test_chunk_documents(self):
        from langchain_core.documents import Document

        long_text = "This is a sentence. " * 200
        docs = [Document(page_content=long_text, metadata={"source": "test"})]
        chunks = chunk_documents(docs)
        assert len(chunks) > 1
        assert all("chunk_index" in c.metadata for c in chunks)

    def test_empty_input(self):
        chunks = chunk_documents([])
        assert chunks == []
