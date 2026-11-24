"""
Model registry — loading, caching, and managing ML models.

Handles LLM, embedding, and reranker model lifecycle.
Models are registered in the ``model_registry`` DB table.
All inference is performed via external vLLM HTTP servers
(see docker-compose.yml for service definitions).
"""

from __future__ import annotations

import logging
from typing import Callable

from sqlalchemy.orm import Session

from app.core.config import get
from app.db.models import ModelRegistry

logger = logging.getLogger(__name__)


def register_model(
    db: Session,
    name: str,
    model_type: str,
    model_path: str,
    uploaded_by: str,
    vector_dimension: int | None = None,
    max_tokens: int | None = None,
    is_default: bool = False,
    api_endpoint: str | None = None,
    config: dict | None = None,
) -> ModelRegistry:
    """Register a new model in the database."""
    import json

    # If setting as default, un-default existing models of same type
    if is_default:
        db.query(ModelRegistry).filter(
            ModelRegistry.model_type == model_type,
            ModelRegistry.is_default == True,
        ).update({"is_default": False})

    model = ModelRegistry(
        name=name,
        model_type=model_type,
        model_path=model_path,
        api_endpoint=api_endpoint,
        vector_dimension=vector_dimension,
        max_tokens=max_tokens,
        is_default=is_default,
        config_json=json.dumps(config) if config else None,
        uploaded_by=uploaded_by,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    logger.info("Registered model '%s' (type=%s, path=%s)", name, model_type, model_path)
    return model


def list_models(db: Session, model_type: str | None = None) -> list[ModelRegistry]:
    q = db.query(ModelRegistry)
    if model_type:
        q = q.filter(ModelRegistry.model_type == model_type)
    return q.order_by(ModelRegistry.model_type, ModelRegistry.name).all()


def get_model(db: Session, model_id: str) -> ModelRegistry | None:
    return db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()


def get_default_model(db: Session, model_type: str) -> ModelRegistry | None:
    return (
        db.query(ModelRegistry)
        .filter(ModelRegistry.model_type == model_type, ModelRegistry.is_default == True)
        .first()
    )


def set_default_model(db: Session, model_id: str) -> ModelRegistry | None:
    model = get_model(db, model_id)
    if model is None:
        return None
    # Un-default others of same type
    db.query(ModelRegistry).filter(
        ModelRegistry.model_type == model.model_type,
        ModelRegistry.is_default == True,
    ).update({"is_default": False})
    model.is_default = True
    db.commit()
    db.refresh(model)
    return model


def delete_model(db: Session, model_id: str) -> bool:
    model = get_model(db, model_id)
    if model is None:
        return False
    db.delete(model)
    db.commit()
    return True


# ── Embedding via vLLM HTTP API ───────────────────────────────────

def _encode(texts: list[str], model_name: str) -> list[list[float]]:
    """Call the vLLM embedding server's OpenAI-compatible endpoint.

    Uses ``/v1/embeddings`` as validated in tests/stream_test.ipynb.
    """
    import requests

    base_url = get("models.embedding_api_endpoint", "http://localhost:8002")
    url = f"{base_url}/v1/embeddings"

    payload = {
        "model": model_name,
        "input": texts,
        "encoding_format": "float",
    }

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # Sort embeddings by index to guarantee order
        embeddings_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in embeddings_data]
    except Exception as e:
        logger.error("Failed to get embeddings from vLLM at %s: %s", url, e)
        raise


def encode_text(text: str, model_path: str | None = None) -> list[float]:
    """Encode a single text string into a dense vector via vLLM."""
    path = model_path or get("models.default_embedding", "Qwen3-VL-Embedding-2B")
    return _encode([text], path)[0]


def encode_batch(
    texts: list[str],
    model_path: str | None = None,
    batch_size: int = 16,
) -> list[list[float]]:
    """Encode a batch of texts into dense vectors via vLLM.

    Splits into sub-batches to avoid oversized HTTP payloads.
    """
    path = model_path or get("models.default_embedding", "Qwen3-VL-Embedding-2B")
    results: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        results.extend(_encode(texts[i : i + batch_size], path))
    return results


def get_embedding_function(
    db: Session,
    embedding_model_id: str | None = None,
) -> Callable[[str], list[float]] | None:
    """Return a callable that encodes text using the specified embedding model.

    Used by the RAG pipeline to encode queries.
    """
    if embedding_model_id:
        model = get_model(db, embedding_model_id)
        if model and model.model_path:
            return lambda text: encode_text(text, model.model_path)

    # Fall back to default
    default = get_default_model(db, "embedding")
    if default and default.model_path:
        return lambda text: encode_text(text, default.model_path)

    # Fall back to config
    default_path = get("models.default_embedding")
    if default_path:
        return lambda text: encode_text(text, default_path)

    return None


def unload_embedding_model() -> None:
    """No-op: embedding model is served externally via vLLM."""
    logger.info("Embedding model is served externally via vLLM; nothing to unload.")
