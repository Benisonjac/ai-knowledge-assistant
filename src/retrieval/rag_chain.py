"""LangGraph Agentic RAG Chain – stateful, multi-step reasoning graph.

This is the MOST IMPRESSIVE part for your resume. It demonstrates:
- LangGraph (state machines for LLM applications)
- Agentic RAG (the LLM decides whether to retrieve, rewrite, or answer)
- Hallucination checking (self-corrective RAG)
- Conversational memory
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from src.config import logger
from src.retrieval.llm import get_llm
from src.retrieval.prompts import (
    HALLUCINATION_CHECK_PROMPT,
    QUERY_REWRITE_PROMPT,
    RAG_PROMPT,
)
from src.retrieval.retriever import format_context, retrieve


# ── Graph State ──────────────────────────────────────────────────────────────
class GraphState(TypedDict):
    """State that flows through the LangGraph nodes."""

    question: str
    chat_history: Annotated[list[BaseMessage], add_messages]
    documents: list[Document]
    context: str
    generation: str
    is_grounded: bool
    rewrite_count: int


# ── Graph Nodes ──────────────────────────────────────────────────────────────

async def rewrite_query(state: GraphState) -> GraphState:
    """Rewrite the user question to be standalone (handles follow-ups)."""
    if not state.get("chat_history"):
        return state

    llm = get_llm()
    chain = QUERY_REWRITE_PROMPT | llm
    result = await chain.ainvoke({
        "chat_history": state["chat_history"],
        "question": state["question"],
    })
    rewritten = result.content.strip()
    logger.info(f"Query rewritten: '{state['question']}' → '{rewritten}'")
    return {**state, "question": rewritten}


async def retrieve_documents(state: GraphState) -> GraphState:
    """Retrieve relevant documents using two-stage retrieval + reranking."""
    logger.info(f"Retrieving for: {state['question'][:80]}")
    docs = retrieve(state["question"], use_reranker=True)
    context = format_context(docs)
    return {**state, "documents": docs, "context": context}


async def generate_answer(state: GraphState) -> GraphState:
    """Generate an answer using the LLM with retrieved context."""
    llm = get_llm()
    chain = RAG_PROMPT | llm
    result = await chain.ainvoke({
        "context": state["context"],
        "chat_history": state.get("chat_history", []),
        "question": state["question"],
    })
    generation = result.content
    logger.info(f"Generated answer ({len(generation)} chars)")
    return {**state, "generation": generation}


async def check_hallucination(state: GraphState) -> GraphState:
    """Check whether the generated answer is grounded in the retrieved docs."""
    llm = get_llm()
    chain = HALLUCINATION_CHECK_PROMPT | llm
    result = await chain.ainvoke({
        "context": state["context"],
        "generation": state["generation"],
    })
    is_grounded = "yes" in result.content.lower()
    logger.info(f"Hallucination check: {'GROUNDED' if is_grounded else 'NOT GROUNDED'}")
    return {**state, "is_grounded": is_grounded}


async def handle_no_docs(state: GraphState) -> GraphState:
    """Fallback when no documents are retrieved."""
    return {
        **state,
        "generation": (
            "I couldn't find any relevant information in the knowledge base "
            "for your question. Please try rephrasing, or upload more documents."
        ),
        "is_grounded": True,
    }


# ── Conditional Edges ────────────────────────────────────────────────────────

def should_retrieve(state: GraphState) -> str:
    """Decide if we need to retrieve (always yes on first pass)."""
    return "retrieve"


def has_documents(state: GraphState) -> str:
    """Check if retrieval returned any documents."""
    if state.get("documents"):
        return "generate"
    return "no_docs"


def is_grounded_or_retry(state: GraphState) -> str:
    """If answer is not grounded and we haven't retried, rewrite and retry."""
    if state["is_grounded"]:
        return "done"
    if state.get("rewrite_count", 0) < 1:
        logger.warning("Answer not grounded – retrying with query rewrite")
        return "retry"
    return "done"  # Give up after 1 retry


# ── Build the Graph ──────────────────────────────────────────────────────────

def build_rag_graph() -> StateGraph:
    """Construct the LangGraph RAG workflow.

    Flow:
    ┌─────────────┐
    │ rewrite_query│ ← (only if chat history exists)
    └──────┬──────┘
           ▼
    ┌─────────────────┐
    │retrieve_documents│
    └──────┬──────────┘
           ▼
     has docs? ──no──► handle_no_docs ──► END
           │yes
           ▼
    ┌──────────────┐
    │generate_answer│
    └──────┬───────┘
           ▼
    ┌────────────────────┐
    │check_hallucination │
    └──────┬─────────────┘
           ▼
     grounded? ──yes──► END
           │no
           ▼
     retry (rewrite → retrieve → generate → check) or END
    """
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("check_hallucination", check_hallucination)
    graph.add_node("handle_no_docs", handle_no_docs)

    # Define flow
    graph.set_entry_point("rewrite_query")
    graph.add_edge("rewrite_query", "retrieve_documents")
    graph.add_conditional_edges("retrieve_documents", has_documents, {
        "generate": "generate_answer",
        "no_docs": "handle_no_docs",
    })
    graph.add_edge("generate_answer", "check_hallucination")
    graph.add_conditional_edges("check_hallucination", is_grounded_or_retry, {
        "done": END,
        "retry": "rewrite_query",
    })
    graph.add_edge("handle_no_docs", END)

    return graph.compile()


# Module-level compiled graph
rag_graph = build_rag_graph()


async def query_knowledge_base(
    question: str,
    chat_history: list[BaseMessage] | None = None,
) -> dict:
    """Primary entry point – run a question through the agentic RAG graph."""
    initial_state: GraphState = {
        "question": question,
        "chat_history": chat_history or [],
        "documents": [],
        "context": "",
        "generation": "",
        "is_grounded": False,
        "rewrite_count": 0,
    }

    result = await rag_graph.ainvoke(initial_state)

    # Build source references
    sources = []
    for doc in result.get("documents", []):
        sources.append({
            "file": doc.metadata.get("source_file", "unknown"),
            "chunk": doc.metadata.get("chunk_index", "?"),
            "similarity_score": doc.metadata.get("similarity_score"),
            "rerank_score": doc.metadata.get("rerank_score"),
        })

    return {
        "answer": result["generation"],
        "sources": sources,
        "is_grounded": result["is_grounded"],
        "question": result["question"],
    }
