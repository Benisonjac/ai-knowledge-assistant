"""RAG Evaluation using RAGAS – quantitative quality metrics.

This module shows you understand how to MEASURE RAG quality, not just build it.
Evaluates: Faithfulness, Answer Relevancy, Context Precision, Context Recall.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from src.config import get_settings, logger
from src.ingestion.embeddings import get_embedding_model


@dataclass
class EvalSample:
    """A single evaluation sample."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str  # Expected answer for reference-based metrics


async def evaluate_rag_pipeline(samples: list[EvalSample]) -> dict:
    """Run RAGAS evaluation on RAG output samples.

    Metrics:
    - Faithfulness: Is the answer grounded in the context? (0-1)
    - Answer Relevancy: Does the answer actually address the question? (0-1)
    - Context Precision: Are retrieved docs relevant to the question? (0-1)
    - Context Recall: Did we retrieve all needed information? (0-1)
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from datasets import Dataset
    except ImportError:
        logger.warning("RAGAS or datasets not installed – skipping evaluation")
        return {"error": "ragas not installed"}

    cfg = get_settings()

    # Convert to RAGAS dataset format
    data = {
        "question": [s.question for s in samples],
        "answer": [s.answer for s in samples],
        "contexts": [s.contexts for s in samples],
        "ground_truth": [s.ground_truth for s in samples],
    }

    dataset = Dataset.from_dict(data)

    # Run evaluation
    logger.info(f"Running RAGAS evaluation on {len(samples)} samples...")
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=cfg.llm_model_name,
        task="text-generation",
        max_new_tokens=cfg.llm_max_tokens,
        temperature=cfg.llm_temperature,
        huggingfacehub_api_token=cfg.huggingfacehub_api_token,
    )
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ChatHuggingFace(llm=llm_endpoint),
        embeddings=get_embedding_model(),
    )

    scores = {
        "faithfulness": round(result["faithfulness"], 4),
        "answer_relevancy": round(result["answer_relevancy"], 4),
        "context_precision": round(result["context_precision"], 4),
        "context_recall": round(result["context_recall"], 4),
    }

    logger.info(f"RAGAS Evaluation Results: {scores}")
    return scores


# ── Sample Evaluation Runner ─────────────────────────────────────────────────

SAMPLE_EVAL_DATA = [
    EvalSample(
        question="What is retrieval augmented generation?",
        answer=(
            "Retrieval Augmented Generation (RAG) is a technique that enhances LLM "
            "responses by first retrieving relevant documents from a knowledge base, "
            "then using those documents as context for the LLM to generate answers."
        ),
        contexts=[
            "RAG combines information retrieval with text generation. It first retrieves "
            "relevant documents using vector similarity search, then feeds them as context "
            "to a large language model to generate accurate, grounded responses."
        ],
        ground_truth=(
            "RAG is a technique that retrieves relevant documents from a knowledge base "
            "and uses them as context for LLM generation, improving accuracy and reducing "
            "hallucinations."
        ),
    ),
]
