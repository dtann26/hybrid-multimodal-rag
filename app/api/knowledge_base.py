"""
Knowledge Base API — CRUD for knowledge bases and folders.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_admin, get_db
from app.db.models import User
from app.services.kb_service import (
    create_folder,
    create_knowledge_base,
    delete_folder,
    delete_knowledge_base,
    get_kb_stats,
    get_knowledge_base,
    list_folders,
    list_knowledge_bases,
    update_knowledge_base,
)

router = APIRouter()


# ── Schemas ─────────────────────────────────────────────────────────

class CreateKBRequest(BaseModel):
    name: str
    description: str | None = None
    is_public: bool = True
    embedding_model_id: str | None = None


class UpdateKBRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    is_public: bool | None = None


class CreateFolderRequest(BaseModel):
    name: str
    description: str | None = None
    parent_id: str | None = None


# ── Knowledge Base endpoints ────────────────────────────────────────

@router.get("/")
def api_list_kbs(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    kbs = list_knowledge_bases(db, user)
    return [
        {
            "id": kb.id,
            "name": kb.name,
            "description": kb.description,
            "is_public": kb.is_public,
            "collection_name": kb.collection_name,
            "vector_dimension": kb.vector_dimension,
            "created_at": kb.created_at.isoformat() if kb.created_at else None,
        }
        for kb in kbs
    ]


@router.post("/", status_code=201)
def api_create_kb(
    body: CreateKBRequest,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    kb = create_knowledge_base(
        db,
        name=body.name,
        created_by=user.id,
        description=body.description,
        is_public=body.is_public,
        embedding_model_id=body.embedding_model_id,
    )
    return {"id": kb.id, "name": kb.name, "collection_name": kb.collection_name}


@router.get("/{kb_id}")
def api_get_kb(
    kb_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    kb = get_knowledge_base(db, kb_id)
    if kb is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return {
        "id": kb.id,
        "name": kb.name,
        "description": kb.description,
        "is_public": kb.is_public,
        "collection_name": kb.collection_name,
        "vector_dimension": kb.vector_dimension,
        "embedding_model_id": kb.embedding_model_id,
        "created_at": kb.created_at.isoformat() if kb.created_at else None,
    }


@router.get("/{kb_id}/stats")
def api_kb_stats(
    kb_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    stats = get_kb_stats(db, kb_id)
    if not stats:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return stats


@router.put("/{kb_id}")
def api_update_kb(
    kb_id: str,
    body: UpdateKBRequest,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    updates = body.model_dump(exclude_none=True)
    kb = update_knowledge_base(db, kb_id, **updates)
    if kb is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return {"id": kb.id, "name": kb.name}


@router.delete("/{kb_id}")
def api_delete_kb(
    kb_id: str,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    ok = delete_knowledge_base(db, kb_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return {"ok": True}


# ── Folder endpoints ───────────────────────────────────────────────

@router.get("/{kb_id}/folders")
def api_list_folders(
    kb_id: str,
    parent_id: str | None = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    folders = list_folders(db, kb_id, parent_id)
    return [
        {
            "id": f.id,
            "name": f.name,
            "description": f.description,
            "parent_id": f.parent_id,
        }
        for f in folders
    ]


@router.post("/{kb_id}/folders", status_code=201)
def api_create_folder(
    kb_id: str,
    body: CreateFolderRequest,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    folder = create_folder(db, kb_id, body.name, body.parent_id, body.description)
    return {"id": folder.id, "name": folder.name}


@router.delete("/folders/{folder_id}")
def api_delete_folder(
    folder_id: str,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    ok = delete_folder(db, folder_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Folder not found")
    return {"ok": True}
