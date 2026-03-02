"""Streamlit Cloud entry-point – standalone app that calls the RAG pipeline directly.

This file is designed for deployment on Streamlit Community Cloud where FastAPI
cannot run alongside. It imports the RAG pipeline, ingestion, and vector store
modules directly instead of calling them via HTTP.

For local development with the split FastAPI + Streamlit setup, use:
    streamlit run src/ui/chat_app.py
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import streamlit as st

# ── Inject secrets into env BEFORE importing src modules ─────────────────────
# Streamlit Cloud stores secrets in st.secrets; our pydantic-settings reads env vars.
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
if "LANGCHAIN_API_KEY" in st.secrets:
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Now safe to import the project modules
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from src.config import logger  # noqa: E402
from src.ingestion import ingest_directory, ingest_file, get_store_stats  # noqa: E402
from src.retrieval import query_knowledge_base  # noqa: E402


# ── Helpers ──────────────────────────────────────────────────────────────────
def run_async(coro):
    """Run an async coroutine from synchronous Streamlit code."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global polish ─────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Hide default Streamlit branding ──────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Sidebar styling ─────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1923 0%, #1a1f2e 100%);
        border-right: 1px solid rgba(79, 139, 249, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown h1 {
        background: linear-gradient(135deg, #4F8BF9 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(79, 139, 249, 0.12);
        margin: 0.75rem 0;
    }

    /* ── Chat header area ────────────────────────────────── */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #4F8BF9 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .main-header p {
        color: #8b95a5;
        font-size: 0.9rem;
        margin: 0;
    }

    /* ── Empty state ─────────────────────────────────────── */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 4rem 2rem;
        text-align: center;
    }
    .empty-state .icon { font-size: 3rem; margin-bottom: 1rem; opacity: 0.7; }
    .empty-state h3 {
        color: #c9d1d9;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .empty-state p { color: #6b7685; font-size: 0.9rem; max-width: 32rem; }
    .suggestion-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        justify-content: center;
        margin-top: 1.25rem;
    }
    .chip {
        background: rgba(79, 139, 249, 0.08);
        border: 1px solid rgba(79, 139, 249, 0.2);
        border-radius: 2rem;
        padding: 0.45rem 1rem;
        font-size: 0.82rem;
        color: #93a3b8;
        cursor: default;
        transition: all 0.2s ease;
    }
    .chip:hover { background: rgba(79, 139, 249, 0.15); color: #c5d0de; }

    /* ── Source cards ─────────────────────────────────────── */
    .source-card {
        background: rgba(79, 139, 249, 0.04);
        border: 1px solid rgba(79, 139, 249, 0.12);
        border-radius: 0.5rem;
        padding: 0.6rem 0.85rem;
        margin-bottom: 0.4rem;
        font-size: 0.82rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .source-card .file-icon { font-size: 1rem; opacity: 0.7; }
    .source-card .file-name { font-weight: 600; color: #c9d1d9; }
    .source-card .meta { color: #6b7685; font-size: 0.75rem; }

    /* ── Grounded badge ──────────────────────────────────── */
    .grounded-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.3rem 0.75rem;
        border-radius: 2rem;
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .grounded-yes {
        background: rgba(52, 211, 153, 0.1);
        border: 1px solid rgba(52, 211, 153, 0.3);
        color: #34d399;
    }
    .grounded-no {
        background: rgba(251, 191, 36, 0.1);
        border: 1px solid rgba(251, 191, 36, 0.3);
        color: #fbbf24;
    }

    /* ── Sidebar section labels ──────────────────────────── */
    .sidebar-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #4F8BF9;
        margin-bottom: 0.4rem;
    }

    /* ── Sidebar stat card ───────────────────────────────── */
    .stat-card {
        background: rgba(79, 139, 249, 0.06);
        border: 1px solid rgba(79, 139, 249, 0.12);
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        text-align: center;
    }
    .stat-card .number {
        font-size: 1.75rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4F8BF9, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-card .label { font-size: 0.72rem; color: #6b7685; margin-top: 0.1rem; }

    /* ── Tech footer pill ────────────────────────────────── */
    .tech-footer {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 0.5rem;
        padding: 0.65rem 0.85rem;
        font-size: 0.72rem;
        color: #555e6e;
        line-height: 1.6;
        text-align: center;
    }

    /* ── Streamlit element tweaks ─────────────────────────── */
    .stChatInput > div { border-radius: 0.75rem !important; }
    div[data-testid="stExpander"] {
        border: 1px solid rgba(79, 139, 249, 0.1) !important;
        border-radius: 0.5rem !important;
        background: rgba(0,0,0,0.15) !important;
    }
    .stButton > button {
        border-radius: 0.5rem;
        font-weight: 500;
        font-size: 0.85rem;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper: render source cards ──────────────────────────────────────────────
def _render_sources(sources: list[dict], is_grounded: bool | None = None):
    """Render professional source cards and grounded badge."""
    for src in sources:
        fname = src.get("file", "unknown")
        chunk = src.get("chunk", "?")
        sim = src.get("similarity_score")
        rerank = src.get("rerank_score")
        sim_str = f"{sim:.3f}" if isinstance(sim, (int, float)) else "—"
        rerank_str = f"{rerank:.3f}" if isinstance(rerank, (int, float)) else "—"
        st.markdown(
            f'<div class="source-card">'
            f'<span class="file-icon">📄</span>'
            f'<span><span class="file-name">{fname}</span> '
            f'<span class="meta"> &nbsp;Chunk {chunk} &nbsp;·&nbsp; '
            f'Sim {sim_str} &nbsp;·&nbsp; Rerank {rerank_str}</span></span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    if is_grounded is not None:
        cls = "grounded-yes" if is_grounded else "grounded-no"
        icon = "✓ Grounded" if is_grounded else "⚠ Not grounded"
        st.markdown(
            f'<span class="grounded-badge {cls}">{icon}</span>',
            unsafe_allow_html=True,
        )


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 AI Knowledge Assistant")
    st.markdown("---")

    # ── Upload ──
    st.markdown('<p class="sidebar-label">Upload Documents</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "PDF, DOCX, TXT, MD, or PPTX",
        type=["pdf", "docx", "txt", "md", "pptx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and st.button("🚀 Ingest Documents", type="primary", use_container_width=True):
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}…"):
                try:
                    suffix = Path(uploaded_file.name).suffix.lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = Path(tmp.name)

                    result = ingest_file(tmp_path)
                    tmp_path.unlink(missing_ok=True)

                    if result.get("status") == "success":
                        st.success(f"{uploaded_file.name} — {result['chunks_added']} chunks")
                    else:
                        st.error(f"{uploaded_file.name}: {result.get('status')}")
                except Exception as e:
                    st.error(f"{uploaded_file.name}: {e}")

    st.markdown("---")

    # ── Directory Ingest ──
    st.markdown('<p class="sidebar-label">Ingest Directory</p>', unsafe_allow_html=True)
    dir_path = st.text_input("Path", value="./data/documents", label_visibility="collapsed")
    if st.button("📥 Ingest Directory", use_container_width=True):
        with st.spinner("Ingesting…"):
            try:
                directory = Path(dir_path)
                if not directory.exists():
                    st.error(f"Directory not found: {dir_path}")
                else:
                    result = ingest_directory(directory)
                    st.success(
                        f"{result['documents_loaded']} docs → {result['chunks_added']} chunks"
                    )
            except Exception as e:
                st.error(str(e))

    st.markdown("---")

    # ── Stats ──
    st.markdown('<p class="sidebar-label">Knowledge Base</p>', unsafe_allow_html=True)
    if st.button("🔄 Refresh Stats", use_container_width=True):
        try:
            stats = get_store_stats()
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="number">{stats["total_chunks"]}</div>'
                f'<div class="label">chunks indexed · {stats["collection_name"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(str(e))

    st.markdown("---")

    # ── Clear ──
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<div class="tech-footer">'
        "LangChain · LangGraph · ChromaDB · HuggingFace · Streamlit<br>"
        "Agentic RAG · Reranking · Hallucination Checking · Observability"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Initialize Session State ────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ── Main Chat Area ───────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-header">'
    "<h1>AI Knowledge Assistant</h1>"
    "<p>Ask anything about your documents — powered by Agentic RAG</p>"
    "</div>",
    unsafe_allow_html=True,
)

# Empty state
if not st.session_state.messages:
    st.markdown(
        '<div class="empty-state">'
        '<div class="icon">💬</div>'
        "<h3>Start a conversation</h3>"
        "<p>Upload documents in the sidebar, then ask questions. "
        "The assistant retrieves relevant passages, reranks them, and generates "
        "a grounded answer with source citations.</p>"
        '<div class="suggestion-chips">'
        '<span class="chip">What is retrieval augmented generation?</span>'
        '<span class="chip">Summarize the key concepts</span>'
        '<span class="chip">How do embeddings work?</span>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources & Retrieval Scores", expanded=False):
                _render_sources(message["sources"], message.get("is_grounded"))


# ── Chat Input ───────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build LangChain message history
    lc_history = []
    for msg in st.session_state.chat_history:
        if msg["role"] == "human":
            lc_history.append(HumanMessage(content=msg["content"]))
        else:
            lc_history.append(AIMessage(content=msg["content"]))

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base & generating answer…"):
            try:
                result = run_async(
                    query_knowledge_base(
                        question=prompt,
                        chat_history=lc_history if lc_history else None,
                    )
                )

                answer = result["answer"]
                sources = result.get("sources", [])
                is_grounded = result.get("is_grounded", True)

                st.markdown(answer)
                if sources:
                    with st.expander("📚 Sources & Retrieval Scores", expanded=False):
                        _render_sources(sources, is_grounded)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "is_grounded": is_grounded,
                })
                st.session_state.chat_history.append(
                    {"role": "human", "content": prompt}
                )
                st.session_state.chat_history.append(
                    {"role": "ai", "content": answer}
                )

            except Exception as e:
                logger.error(f"Query failed: {e}")
                st.error(f"Error: {e}")
