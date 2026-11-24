"""
Reranker module — re-scores retrieved chunks using a cross-encoder model.

The reranker runs as an external vLLM service (see docker-compose.yml).
Default model: Qwen3-VL-Reranker-2B (configurable via ModelRegistry).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def rerank(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int = 5,
    model_path: str | None = None,
) -> list[dict[str, Any]]:
    """Re-score *chunks* by relevance to *query* via the vLLM reranker API.

    Each chunk dict must have a ``"content"`` key.
    A ``"rerank_score"`` key is added to the returned dicts.

    Uses the ``/rerank`` endpoint as validated in tests/stream_test.ipynb.
    """
    if not chunks:
        return []

    import requests
    from app.core.config import get

    base_url = get("models.reranker_api_endpoint", "http://localhost:8003")
    path = model_path or get("models.default_reranker", "Qwen3-VL-Reranker-2B")
    url = f"{base_url}/rerank"

    documents = [c["content"] for c in chunks]

    payload = {
        "model": path,
        "query": query,
        "documents": documents,
        "top_n": top_k,
    }

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()

        # Response format from vLLM reranker API:
        # { "results": [ {"index": 0, "relevance_score": 0.8, ...}, ... ] }
        data = resp.json()

        # Attach scores to the original chunks
        for item in data.get("results", []):
            idx = item.get("index")
            score = item.get("relevance_score")
            if idx is not None and idx < len(chunks):
                chunks[idx]["rerank_score"] = score

        # Chunks not returned by the API (filtered by top_n) get a low default
        for c in chunks:
            if "rerank_score" not in c:
                c["rerank_score"] = -999.0

        ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_k]
    except Exception as e:
        logger.error("Failed to rerank via vLLM at %s: %s. Returning chunks as-is.", url, e)
        return chunks[:top_k]


def unload_reranker() -> None:
    """No-op: reranker is served externally via vLLM."""
    logger.info("Reranker is served externally via vLLM; nothing to unload.")
