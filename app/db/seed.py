"""
Database seeding — creates the superadmin user, default models,
and the "General" default knowledge base on first startup.

Backward-compatible: if the DB already exists (with or without the
"General" KB), no existing data is modified or destroyed.
"""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app.core.config import get
from app.db.models import KnowledgeBase, ModelRegistry, User
from app.services.auth_service import create_user, get_user_by_username
from app.models_manager.registry import register_model
from app.services.kb_service import create_knowledge_base

logger = logging.getLogger(__name__)


def seed_superadmin(db: Session) -> User:
    """Create the superadmin user if it doesn't exist."""
    username = get("superadmin.username", "admin")
    existing = get_user_by_username(db, username)
    if existing:
        logger.info("Superadmin '%s' already exists", username)
        return existing

    user = create_user(
        db,
        username=username,
        email=get("superadmin.email", "admin@localhost"),
        password=get("superadmin.password", "admin123"),
        role="superadmin",
        display_name="Super Admin",
    )
    logger.info("Created superadmin user '%s'", username)
    return user


def seed_default_models(db: Session, admin_id: str) -> None:
    """Register the hardcoded LLM, embedding, and reranker models.

    LLM models are strictly limited to these three:
      1. Qwen2.5-VL-7B-Instruct-NVFP4  (default)
      2. Phi-4-multimodal-instruct-NVFP4
      3. Gemma-4-31B-IT-NVFP4
    """
    defaults = [
        # ── LLM models (hardcoded) ──────────────────────────────────
        {
            "name": "Qwen2.5-VL-7B-Instruct-NVFP4",
            "model_type": "llm",
            "model_path": "Qwen2.5-VL-7B-Instruct-NVFP4",
            "max_tokens": 32768,
            "is_default": True,
        },
        {
            "name": "Phi-4-multimodal-instruct-NVFP4",
            "model_type": "llm",
            "model_path": "Phi-4-multimodal-instruct-NVFP4",
            "max_tokens": 32768,
            "is_default": False,
        },
        {
            "name": "Gemma-4-31B-IT-NVFP4",
            "model_type": "llm",
            "model_path": "Gemma-4-31B-IT-NVFP4",
            "max_tokens": 32768,
            "is_default": False,
        },
        # ── Embedding & reranker ────────────────────────────────────
        {
            "name": "Qwen3-VL-Embedding-2B",
            "model_type": "embedding",
            "model_path": get("models.default_embedding", "Qwen3-VL-Embedding-2B"),
            "vector_dimension": get("models.embedding_dimension", 2048),
            "is_default": True,
        },
        {
            "name": "Qwen3-VL-Reranker-2B",
            "model_type": "reranker",
            "model_path": get("models.default_reranker", "Qwen3-VL-Reranker-2B"),
            "is_default": True,
        },
    ]

    for model_info in defaults:
        existing = (
            db.query(ModelRegistry)
            .filter(ModelRegistry.model_path == model_info["model_path"])
            .first()
        )
        if existing:
            logger.info("Model '%s' already registered", model_info["name"])
            continue

        register_model(
            db=db,
            uploaded_by=admin_id,
            **model_info,
        )
        logger.info("Registered default model: %s", model_info["name"])


def seed_default_kb(db: Session, admin_id: str) -> KnowledgeBase | None:
    """Create the 'General' default knowledge base if it doesn't exist.

    Backward-compatible: skips silently if a KB named 'General' already
    exists so that re-running seed on an existing database is safe.
    """
    existing = db.query(KnowledgeBase).filter(KnowledgeBase.name == "General").first()
    if existing:
        logger.info("KB 'General' already exists (id=%s)", existing.id)
        return existing

    # Get default embedding model (may be None on a fresh install before models are loaded)
    embed_model = (
        db.query(ModelRegistry)
        .filter(ModelRegistry.model_type == "embedding", ModelRegistry.is_default == True)
        .first()
    )

    kb = create_knowledge_base(
        db=db,
        name="General",
        created_by=admin_id,
        description="Default general-purpose knowledge base.",
        is_public=True,
        embedding_model_id=embed_model.id if embed_model else None,
    )
    logger.info("Created default KB 'General' (id=%s)", kb.id)
    return kb


def run_seed(db: Session) -> None:
    """Run all seeding operations."""
    logger.info("Running database seed...")
    admin = seed_superadmin(db)
    seed_default_models(db, admin.id)
    seed_default_kb(db, admin.id)
    logger.info("Database seed complete.")
