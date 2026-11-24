"""
Knowledge Base service — CRUD for knowledge bases and folders.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import get
from app.db.models import Folder, KnowledgeBase, ModelRegistry, Role, RoleKBAccess, User
from app.rag.retriever import create_collection, drop_collection, collection_exists
from app.services.document_store import (
    count_documents,
    count_documents_by_status,
    delete_documents_in_folder,
)

logger = logging.getLogger(__name__)


def _make_collection_name(kb_name: str, kb_id: str) -> str:
    """Generate a Milvus-safe collection name from KB name + id prefix."""
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", kb_name).lower()
    return f"kb_{safe}_{kb_id[:8]}"


# ── Knowledge Base CRUD ─────────────────────────────────────────────

def create_knowledge_base(
    db: Session,
    name: str,
    created_by: str,
    description: str | None = None,
    is_public: bool = True,
    embedding_model_id: str | None = None,
) -> KnowledgeBase:
    # Resolve vector dimension from embedding model
    vector_dim = get("models.embedding_dimension", 2048)
    if embedding_model_id:
        model = db.query(ModelRegistry).filter(ModelRegistry.id == embedding_model_id).first()
        if model and model.vector_dimension:
            vector_dim = model.vector_dimension

    kb = KnowledgeBase(
        name=name,
        description=description,
        is_public=is_public,
        created_by=created_by,
        embedding_model_id=embedding_model_id,
        vector_dimension=vector_dim,
    )
    db.add(kb)
    db.flush()  # get the id

    # Create Milvus collection
    collection_name = _make_collection_name(name, kb.id)
    kb.collection_name = collection_name
    create_collection(collection_name, vector_dim)

    # Create default root folder
    root_folder = Folder(
        knowledge_base_id=kb.id,
        name="Documents",
        description="Default folder",
    )
    db.add(root_folder)
    db.commit()
    db.refresh(kb)
    logger.info("Created KB '%s' (collection=%s, dim=%d)", name, collection_name, vector_dim)
    return kb


def get_knowledge_base(db: Session, kb_id: str) -> KnowledgeBase | None:
    return db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()


def list_knowledge_bases(db: Session, user: User | None = None) -> list[KnowledgeBase]:
    """List knowledge bases respecting role-based access.

    - superadmin / admin → all KBs
    - user with custom_role → only KBs assigned to that role
    - user without custom_role → public KBs only
    """
    q = db.query(KnowledgeBase)

    if user and user.role in ("admin", "superadmin"):
        pass  # no filter — full access
    elif user and user.custom_role_id:
        allowed_kb_ids = (
            db.query(RoleKBAccess.kb_id)
            .filter(RoleKBAccess.role_id == user.custom_role_id)
            .subquery()
        )
        q = q.filter(KnowledgeBase.id.in_(allowed_kb_ids))
    elif user:
        q = q.filter(KnowledgeBase.is_public == True)
    else:
        q = q.filter(KnowledgeBase.is_public == True)

    return q.order_by(KnowledgeBase.created_at.desc()).all()


def update_knowledge_base(db: Session, kb_id: str, **kwargs) -> KnowledgeBase | None:
    kb = get_knowledge_base(db, kb_id)
    if kb is None:
        return None
    for key, val in kwargs.items():
        if hasattr(kb, key):
            setattr(kb, key, val)
    db.commit()
    db.refresh(kb)
    return kb


def delete_knowledge_base(db: Session, kb_id: str) -> bool:
    kb = get_knowledge_base(db, kb_id)
    if kb is None:
        return False
    # Drop Milvus collection
    if kb.collection_name:
        drop_collection(kb.collection_name)
    # Clean up document files in all folders
    folders = db.query(Folder).filter(Folder.knowledge_base_id == kb_id).all()
    for folder in folders:
        delete_documents_in_folder(db, folder.id)
    db.delete(kb)
    db.commit()
    logger.info("Deleted KB '%s'", kb.name)
    return True


# ── Folder CRUD ─────────────────────────────────────────────────────

def create_folder(
    db: Session,
    kb_id: str,
    name: str,
    parent_id: str | None = None,
    description: str | None = None,
) -> Folder:
    folder = Folder(
        knowledge_base_id=kb_id,
        name=name,
        parent_id=parent_id,
        description=description,
    )
    db.add(folder)
    db.commit()
    db.refresh(folder)
    return folder


def list_folders(db: Session, kb_id: str, parent_id: str | None = None) -> list[Folder]:
    q = db.query(Folder).filter(Folder.knowledge_base_id == kb_id)
    if parent_id:
        q = q.filter(Folder.parent_id == parent_id)
    else:
        q = q.filter(Folder.parent_id == None)
    return q.order_by(Folder.name).all()


def get_folder(db: Session, folder_id: str) -> Folder | None:
    return db.query(Folder).filter(Folder.id == folder_id).first()


def delete_folder(db: Session, folder_id: str) -> bool:
    folder = get_folder(db, folder_id)
    if folder is None:
        return False
    # Clean up document files
    delete_documents_in_folder(db, folder_id)
    db.delete(folder)
    db.commit()
    return True


# ── Stats ───────────────────────────────────────────────────────────

def get_kb_stats(db: Session, kb_id: str) -> dict[str, Any]:
    kb = get_knowledge_base(db, kb_id)
    if kb is None:
        return {}
    folder_ids = [f.id for f in db.query(Folder).filter(Folder.knowledge_base_id == kb_id).all()]
    doc_count = count_documents(db, folder_ids)
    folder_count = len(folder_ids)
    processed = count_documents_by_status(db, folder_ids, "processed")
    return {
        "id": kb.id,
        "name": kb.name,
        "document_count": doc_count,
        "folder_count": folder_count,
        "processed_count": processed,
        "collection_name": kb.collection_name,
        "vector_dimension": kb.vector_dimension,
    }
