"""
Admin API — model management, config, transcript import, user management,
superadmin utilities, LLM hot-swap.

Most endpoints require ``superadmin`` role.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.api.deps import get_db, require_admin, require_superadmin
from app.core.config import get_config, save_config, update_config
from app.db.models import (
    ChatMessage, ChatSession, Folder,
    KnowledgeBase, Role, RoleKBAccess, User,
)
from app.models_manager.registry import (
    delete_model,
    get_default_model,
    list_models,
    register_model,
    set_default_model,
)
from app.services.auth_service import list_users, update_user_role, deactivate_user, delete_user
from app.services.llm_switch_service import (
    ALLOWED_MODELS,
    LLMStatus,
    get_llm_state,
    list_available_models,
    read_current_model_from_env,
    switch_llm_model,
    MODELS_DIR,
)
from app.services.document_store import (
    count_documents,
    list_documents_by_kb,
    get_all_documents,
    update_document_status,
    delete_document,
)
from app.core.limiter import limiter

router = APIRouter()


# ── Schemas ─────────────────────────────────────────────────────────

class RegisterModelRequest(BaseModel):
    name: str
    model_type: str  # "llm" | "embedding" | "reranker"
    model_path: str  # HuggingFace repo ID or local path
    vector_dimension: int | None = None
    max_tokens: int | None = None
    is_default: bool = False
    api_endpoint: str | None = None
    config: dict | None = None


class UpdateConfigRequest(BaseModel):
    updates: dict


class UpdateUserRoleRequest(BaseModel):
    role: str
    custom_role_id: str | None = None


class CreateRoleRequest(BaseModel):
    name: str
    description: str | None = None


class UpdateRoleRequest(BaseModel):
    name: str | None = None
    description: str | None = None


class SetRoleKBAccessRequest(BaseModel):
    kb_ids: list[str]


# ── LLM hot-swap ───────────────────────────────────────────────────

class SwitchLLMRequest(BaseModel):
    model_dir_name: str


@router.get("/models/llm/available")
def api_list_available_llms(
    user: User = Depends(require_admin),
):
    """Return the hardcoded list of supported LLM models.

    Also includes the currently active model name from ``.env``
    and the live switch status.
    """
    available = list_available_models()
    current = read_current_model_from_env()
    state = get_llm_state()

    # For each model, indicate whether it's already downloaded
    models_with_status = []
    for name in available:
        model_path = MODELS_DIR / name
        downloaded = model_path.exists() and any(model_path.iterdir()) if model_path.exists() else False
        models_with_status.append({
            "name": name,
            "downloaded": downloaded,
            "is_active": name == current,
        })

    return {
        "available_models": available,
        "models_detail": models_with_status,
        "current_model": current,
        "status": state["status"],
        "error": state.get("error"),
    }


@router.post("/models/llm/switch")
@limiter.limit("20/minute")
def api_switch_llm(
    request: Request,
    body: SwitchLLMRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(require_superadmin),
):
    """Initiate a live model switch of the vllm-llm container.

    - Validates the target is one of the 3 allowed models.
    - Rejects if a switch is already in progress.
    - The background task downloads the model if needed, rewrites
      ``docker-compose.yml``, and restarts the container.
    """
    target = body.model_dir_name.strip()

    # Strict validation: only the 3 hardcoded models are accepted
    if target not in ALLOWED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{target}' is not supported. "
                   f"Allowed models: {sorted(ALLOWED_MODELS)}",
        )

    state = get_llm_state()
    if state["status"] == LLMStatus.SWITCHING:
        raise HTTPException(
            status_code=409,
            detail=f"A model switch is already in progress → {state.get('target_model')}",
        )

    current = read_current_model_from_env()
    if current == target:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{target}' is already the active model.",
        )

    background_tasks.add_task(switch_llm_model, target)
    return {
        "ok": True,
        "message": f"Model switch initiated: {current} → {target}",
        "status": "switching",
    }


@router.get("/models/llm/status")
def api_llm_switch_status(
    user: User = Depends(require_admin),
):
    """Poll the current state of an LLM model switch."""
    return get_llm_state()


# ── Model management ───────────────────────────────────────────────

@router.get("/models")
def api_list_models(
    model_type: str | None = None,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    models = list_models(db, model_type)
    return [
        {
            "id": m.id,
            "name": m.name,
            "model_type": m.model_type,
            "model_path": m.model_path,
            "vector_dimension": m.vector_dimension,
            "max_tokens": m.max_tokens,
            "status": m.status,
            "is_default": m.is_default,
            "api_endpoint": m.api_endpoint,
            "config": json.loads(m.config_json) if m.config_json else None,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in models
    ]


@router.post("/models", status_code=201)
def api_register_model(
    body: RegisterModelRequest,
    user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    model = register_model(
        db=db,
        name=body.name,
        model_type=body.model_type,
        model_path=body.model_path,
        uploaded_by=user.id,
        vector_dimension=body.vector_dimension,
        max_tokens=body.max_tokens,
        is_default=body.is_default,
        api_endpoint=body.api_endpoint,
        config=body.config,
    )
    return {"id": model.id, "name": model.name, "status": model.status}


@router.put("/models/{model_id}/default")
def api_set_default_model(
    model_id: str,
    user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    model = set_default_model(db, model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"id": model.id, "name": model.name, "is_default": model.is_default}


@router.delete("/models/{model_id}")
def api_delete_model(
    model_id: str,
    user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    ok = delete_model(db, model_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"ok": True}


# ── Config management ──────────────────────────────────────────────

@router.get("/config")
def api_get_config(user: User = Depends(require_admin)):
    return get_config()


@router.put("/config")
def api_update_config(
    body: UpdateConfigRequest,
    user: User = Depends(require_superadmin),
):
    cfg = update_config(body.updates, persist=True)
    return {"ok": True, "config": cfg}


@router.post("/config/export")
def api_export_config(user: User = Depends(require_admin)):
    """Export the current config as JSON."""
    return get_config()


@router.post("/config/reset")
def api_reset_config(user: User = Depends(require_superadmin)):
    """Reload config from disk (discard in-memory changes)."""
    from app.core.config import load_config
    cfg = load_config(reload=True)
    return {"ok": True, "config": cfg}


# ── Role management ────────────────────────────────────────────────

@router.get("/roles")
def api_list_roles(
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    roles = db.query(Role).order_by(Role.name).all()
    result = []
    for r in roles:
        kb_ids = [a.kb_id for a in r.kb_access]
        user_count = db.query(User).filter(User.custom_role_id == r.id).count()
        result.append({
            "id": r.id,
            "name": r.name,
            "description": r.description,
            "kb_ids": kb_ids,
            "user_count": user_count,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })
    return result


@router.post("/roles", status_code=201)
def api_create_role(
    body: CreateRoleRequest,
    user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    existing = db.query(Role).filter(Role.name == body.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Role name already exists")
    role = Role(name=body.name, description=body.description)
    db.add(role)
    db.commit()
    db.refresh(role)
    return {"id": role.id, "name": role.name}


@router.put("/roles/{role_id}")
def api_update_role(
    role_id: str,
    body: UpdateRoleRequest,
    user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    role = db.query(Role).filter(Role.id == role_id).first()
    if role is None:
        raise HTTPException(status_code=404, detail="Role not found")
    if body.name is not None:
        dup = db.query(Role).filter(Role.name == body.name, Role.id != role_id).first()
        if dup:
            raise HTTPException(status_code=400, detail="Role name already exists")
        role.name = body.name
    if body.description is not None:
        role.description = body.description
    db.commit()
    db.refresh(role)
    return {"id": role.id, "name": role.name}


@router.delete("/roles/{role_id}")
def api_delete_role(
    role_id: str,
    user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    role = db.query(Role).filter(Role.id == role_id).first()
    if role is None:
        raise HTTPException(status_code=404, detail="Role not found")
    # Unset custom_role_id for users who have this role
    db.query(User).filter(User.custom_role_id == role_id).update(
        {User.custom_role_id: None}, synchronize_session="fetch"
    )
    db.delete(role)
    db.commit()
    return {"ok": True}


@router.put("/roles/{role_id}/kb-access")
def api_set_role_kb_access(
    role_id: str,
    body: SetRoleKBAccessRequest,
    user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    """Replace the KB access list for a role."""
    role = db.query(Role).filter(Role.id == role_id).first()
    if role is None:
        raise HTTPException(status_code=404, detail="Role not found")
    # Delete existing access entries
    db.query(RoleKBAccess).filter(RoleKBAccess.role_id == role_id).delete()
    # Insert new ones
    for kb_id in body.kb_ids:
        db.add(RoleKBAccess(role_id=role_id, kb_id=kb_id))
    db.commit()
    return {"ok": True, "kb_ids": body.kb_ids}


@router.get("/roles/{role_id}/kb-access")
def api_get_role_kb_access(
    role_id: str,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    role = db.query(Role).filter(Role.id == role_id).first()
    if role is None:
        raise HTTPException(status_code=404, detail="Role not found")
    access = db.query(RoleKBAccess).filter(RoleKBAccess.role_id == role_id).all()
    return [{"kb_id": a.kb_id} for a in access]


# ── Transcript import ──────────────────────────────────────────────

class ImportTranscriptsRequest(BaseModel):
    kb_id: str | None = None
    kb_name: str | None = None
    folder_id: str | None = None
    batch_size: int = 16


@router.post("/import-transcripts")
@limiter.limit("20/minute")
def api_import_transcripts(
    request: Request,
    body: ImportTranscriptsRequest,
    user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    """Import existing transcript JSON files into a knowledge base."""
    from app.ingest.transcript_importer import import_all_transcripts

    try:
        result = import_all_transcripts(
            db=db,
            kb_id=body.kb_id,
            kb_name=body.kb_name,
            folder_id=body.folder_id,
            batch_size=body.batch_size,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── User management ───────────────────────────────────────────────

@router.get("/users")
def api_list_users(
    skip: int = 0,
    limit: int = 50,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return list_users(db, skip=skip, limit=limit)


@router.put("/users/{user_id}/role")
def api_update_user_role(
    user_id: str,
    body: UpdateUserRoleRequest,
    user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    if body.role not in ("user", "admin", "superadmin"):
        raise HTTPException(status_code=400, detail="Invalid role")
    updated = update_user_role(db, user_id, body.role, body.custom_role_id)
    if updated is None:
        raise HTTPException(status_code=404, detail="User not found")
    return {"id": updated.id, "role": updated.role, "custom_role_id": updated.custom_role_id}


@router.delete("/users/{user_id}")
def api_deactivate_user(
    user_id: str,
    user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    if user_id == user.id:
        raise HTTPException(status_code=400, detail="Cannot deactivate yourself")
    ok = deactivate_user(db, user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="User not found")
    return {"ok": True}


@router.delete("/users/{user_id}/delete")
def api_delete_user(
    user_id: str,
    user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    if user_id == user.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    ok = delete_user(db, user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="User not found")
    return {"ok": True}


# ── Superadmin Utilities ───────────────────────────────────────────

@router.get("/utilities/stats")
def api_system_stats(
    user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    """Return system-wide statistics."""
    from app.rag.retriever import ensure_connection, get_collection_stats
    from pymilvus import utility as milvus_util, connections as milvus_conns

    user_count = db.query(func.count(User.id)).scalar()
    kb_count = db.query(func.count(KnowledgeBase.id)).scalar()
    doc_count = count_documents(db)
    session_count = db.query(func.count(ChatSession.id)).scalar()
    message_count = db.query(func.count(ChatMessage.id)).scalar()

    # Milvus status
    milvus_ok = False
    milvus_collections = 0
    total_milvus_chunks = 0
    try:
        ensure_connection()
        milvus_ok = True
        milvus_collections = len(milvus_util.list_collections())
    except Exception:
        pass

    # Per-KB stats
    kb_stats = []
    kbs = db.query(KnowledgeBase).all()
    for kb in kbs:
        folder_ids = [f.id for f in db.query(Folder).filter(Folder.knowledge_base_id == kb.id).all()]
        docs = count_documents(db, folder_ids)
        milvus_entities = 0
        try:
            stats = get_collection_stats(kb.collection_name)
            milvus_entities = stats.get("num_entities", 0)
        except Exception:
            pass
        total_milvus_chunks += milvus_entities
        kb_stats.append({
            "id": kb.id,
            "name": kb.name,
            "collection_name": kb.collection_name,
            "documents": docs,
            "chunks": milvus_entities,
            "milvus_entities": milvus_entities,
        })

    return {
        "users": user_count,
        "knowledge_bases": kb_count,
        "documents": doc_count,
        "chunks": total_milvus_chunks,
        "chat_sessions": session_count,
        "chat_messages": message_count,
        "milvus_connected": milvus_ok,
        "milvus_collections": milvus_collections,
        "kb_details": kb_stats,
    }


class ReindexRequest(BaseModel):
    batch_size: int = 16


@router.post("/utilities/reindex/{kb_id}")
@limiter.limit("20/minute")
def api_reindex_kb(
    request: Request,
    kb_id: str,
    body: ReindexRequest,
    user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    """Drop the Milvus collection for a KB, recreate it, and re-process all documents."""
    from app.rag.retriever import drop_collection, create_collection, get_collection_stats
    from app.ingest.processor import process_document

    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if kb is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    if not kb.collection_name:
        raise HTTPException(status_code=400, detail="KB has no collection configured")

    # 1. Drop and recreate the collection
    drop_collection(kb.collection_name)
    create_collection(kb.collection_name, kb.vector_dimension or 2048)

    # 2. Get all documents from DB for this KB
    folder_ids = [f.id for f in db.query(Folder).filter(Folder.knowledge_base_id == kb_id).all()]
    docs = list_documents_by_kb(db, folder_ids)
    processed_docs = [d for d in docs if d.get("status") in ("processed", "error")]

    total_indexed = 0
    errors = 0

    for doc in processed_docs:
        try:
            ok = process_document(db, doc["id"], folder_id=doc["folder_id"])
            if ok:
                total_indexed += 1
            else:
                errors += 1
        except Exception as e:
            errors += 1
            update_document_status(db, doc["id"], doc["folder_id"], "error", error_message=f"Reindex failed: {e}")

    stats = get_collection_stats(kb.collection_name)
    return {
        "ok": True,
        "kb_name": kb.name,
        "documents_processed": len(processed_docs),
        "chunks_indexed": total_indexed,
        "errors": errors,
        "milvus_entities": stats.get("num_entities", 0),
    }


@router.delete("/utilities/clear-chats")
def api_clear_all_chats(
    user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    """Delete all chat sessions and messages system-wide."""
    msg_count = db.query(func.count(ChatMessage.id)).scalar()
    session_count = db.query(func.count(ChatSession.id)).scalar()
    db.query(ChatMessage).delete()
    db.query(ChatSession).delete()
    db.commit()
    return {"ok": True, "sessions_deleted": session_count, "messages_deleted": msg_count}


@router.delete("/utilities/orphan-chunks")
def api_purge_orphan_data(
    user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    """Check for orphaned documents whose folder no longer exists."""
    all_docs = get_all_documents(db)
    folder_ids_in_db = {f.id for f in db.query(Folder).all()}
    orphan_count = 0
    for doc in all_docs:
        if doc.get("folder_id") not in folder_ids_in_db:
            delete_document(db, doc["id"], doc["folder_id"])
            orphan_count += 1
    return {"ok": True, "orphan_documents_removed": orphan_count}


@router.post("/utilities/rebuild-collection/{kb_id}")
@limiter.limit("20/minute")
def api_rebuild_collection(
    request: Request,
    kb_id: str,
    user: User = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    """Drop and recreate the Milvus collection (without re-embedding).
    Useful if the index got corrupted."""
    from app.rag.retriever import drop_collection, create_collection

    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if kb is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    if not kb.collection_name:
        raise HTTPException(status_code=400, detail="KB has no collection configured")

    drop_collection(kb.collection_name)
    create_collection(kb.collection_name, kb.vector_dimension or 2048)

    # Mark all docs as pending in DB (they need re-embedding)
    folder_ids = [f.id for f in db.query(Folder).filter(Folder.knowledge_base_id == kb_id).all()]
    docs = list_documents_by_kb(db, folder_ids)
    for d in docs:
        update_document_status(db, d["id"], d["folder_id"], "pending")

    return {"ok": True, "kb_name": kb.name, "documents_reset": len(docs)}
