"""Document loaders – supports PDF, DOCX, PPTX, TXT, and Markdown."""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document

from src.config import logger

# Map file extension → LangChain loader class
_LOADER_MAP: dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".pptx": UnstructuredPowerPointLoader,
}

SUPPORTED_EXTENSIONS = set(_LOADER_MAP.keys())


def load_single_document(file_path: Path) -> list[Document]:
    """Load a single file and return a list of LangChain Documents."""
    ext = file_path.suffix.lower()
    loader_cls = _LOADER_MAP.get(ext)
    if loader_cls is None:
        logger.warning(f"Unsupported file type '{ext}' – skipping {file_path.name}")
        return []

    logger.info(f"Loading {file_path.name} with {loader_cls.__name__}")
    loader = loader_cls(str(file_path))
    docs = loader.load()

    # Enrich metadata
    for doc in docs:
        doc.metadata["source_file"] = file_path.name
        doc.metadata["file_type"] = ext

    return docs


def load_documents_from_directory(directory: Path) -> list[Document]:
    """Recursively load all supported documents from *directory*."""
    all_docs: list[Document] = []
    files = sorted(directory.rglob("*"))

    for fp in files:
        if fp.is_file() and fp.suffix.lower() in SUPPORTED_EXTENSIONS:
            all_docs.extend(load_single_document(fp))

    logger.info(f"Loaded {len(all_docs)} document chunks from {directory}")
    return all_docs
