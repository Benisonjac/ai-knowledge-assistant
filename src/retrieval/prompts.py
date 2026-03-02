"""Prompt templates for the RAG system."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ── Main RAG Prompt ──────────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """\
You are an expert AI Knowledge Assistant. Your job is to provide accurate, \
helpful answers based ONLY on the retrieved context below.

RULES:
1. Answer the question using ONLY the provided context.
2. If the context doesn't contain enough information, say so clearly.
3. Cite your sources using [Source N] references.
4. Be concise but thorough. Use bullet points for complex answers.
5. If the user asks a follow-up, use the chat history for continuity.
6. NEVER make up information not present in the context.

RETRIEVED CONTEXT:
{context}
"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


# ── Query Rewriting Prompt (for conversational RAG) ─────────────────────────
QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given the chat history and a follow-up question, rewrite the question "
        "to be a standalone question that captures the full context needed for "
        "retrieval. Return ONLY the rewritten question, nothing else.",
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


# ── Hallucination Grader Prompt ──────────────────────────────────────────────
HALLUCINATION_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a grader assessing whether an LLM generation is grounded in "
        "the retrieved documents. Give a binary 'yes' or 'no' score. "
        "'yes' means the answer is fully supported by the documents.",
    ),
    (
        "human",
        "Retrieved documents:\n{context}\n\nLLM generation:\n{generation}\n\n"
        "Is this generation grounded in the documents? Answer 'yes' or 'no'.",
    ),
])
