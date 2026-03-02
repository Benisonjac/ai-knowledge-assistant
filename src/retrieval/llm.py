"""LLM factory – uses HuggingFace Inference API (free, no local resources needed)."""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings, logger


@lru_cache
def get_llm() -> ChatHuggingFace:
    """Return a cached ChatHuggingFace instance using the free HF Inference API."""
    cfg = get_settings()
    logger.info(
        f"Initializing HuggingFace LLM: {cfg.llm_model_name} (temp={cfg.llm_temperature})"
    )
    endpoint = HuggingFaceEndpoint(
        repo_id=cfg.llm_model_name,
        task="text-generation",
        max_new_tokens=cfg.llm_max_tokens,
        temperature=cfg.llm_temperature,
        huggingfacehub_api_token=cfg.huggingfacehub_api_token,
    )
    return ChatHuggingFace(llm=endpoint)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def invoke_llm_with_retry(chain, inputs: dict) -> str:
    """Invoke an LLM chain with exponential backoff retry."""
    result = await chain.ainvoke(inputs)
    return result.content if hasattr(result, "content") else str(result)
