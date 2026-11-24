"""
Chat API — chat sessions, messages, and RAG-powered conversation.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.db.models import ModelRegistry, User
from app.services.chat_service import (
    chat,
    chat_stream,
    create_session,
    delete_session,
    edit_user_message,
    get_messages,
    get_message_versions,
    get_session,
    list_sessions,
    switch_to_version,
    update_session,
)
from app.services.llm_switch_service import LLMStatus, get_llm_state
from app.core.limiter import limiter

router = APIRouter()


# ── Maintenance mode guardrail ──────────────────────────────────────

def _require_llm_ready() -> None:
    """Block chat requests while the vLLM container is being swapped.

    Raises ``HTTPException(503)`` when the LLM status is SWITCHING so
    that users get a clear maintenance message instead of a 500 error
    from the dead container.
    """
    state = get_llm_state()
    if state["status"] == LLMStatus.SWITCHING:
        raise HTTPException(
            status_code=503,
            detail=(
                "The AI is currently undergoing a live model swap to free "
                "up VRAM. Please wait a few minutes and try again."
            ),
        )


# ── Public models endpoint (any authenticated user) ─────────────────

@router.get("/models")
def api_list_available_models(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List available models for the chat model selector (no admin required)."""
    models = (
        db.query(ModelRegistry)
        .order_by(ModelRegistry.model_type, ModelRegistry.name)
        .all()
    )
    return [
        {
            "id": m.id,
            "name": m.name,
            "model_type": m.model_type,
            "model_path": m.model_path,
            "is_default": m.is_default,
            "vector_dimension": m.vector_dimension,
            "status": m.status,
        }
        for m in models
    ]


# ── Public LLM status (any authenticated user) ─────────────────────

@router.get("/llm-status")
def api_llm_status(user: User = Depends(get_current_user)):
    """Return the current LLM switching state for the chat UI.

    Standard users use this to display the active model name and see
    if the system is in maintenance mode.
    """
    state = get_llm_state()
    return {
        "status": state["status"],
        "current_model": state.get("current_model"),
        "error": state.get("error"),
    }


# ── Schemas ─────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    title: str = "New Chat"
    kb_ids: list[str] = []
    model_id: str | None = None
    system_prompt: str | None = None


class UpdateSessionRequest(BaseModel):
    title: str | None = None
    kb_ids: list[str] | None = None
    model_id: str | None = None
    system_prompt: str | None = None


class ChatRequest(BaseModel):
    message: str
    images: list[str] | None = None
    message_id: str | None = None
    metadata_filters: dict | None = None



class EditMessageRequest(BaseModel):
    content: str


class SessionResponse(BaseModel):
    id: str
    title: str
    knowledge_base_ids: str | None
    model_id: str | None
    created_at: str

    class Config:
        from_attributes = True


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    sources_json: str | None
    created_at: str

    class Config:
        from_attributes = True


# ── Session endpoints ───────────────────────────────────────────────

@router.get("/sessions")
def api_list_sessions(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    sessions = list_sessions(db, user.id)
    return [
        {
            "id": s.id,
            "title": s.title,
            "kb_ids": json.loads(s.knowledge_base_ids) if s.knowledge_base_ids else [],
            "model_id": s.model_id,
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "updated_at": s.updated_at.isoformat() if s.updated_at else None,
        }
        for s in sessions
    ]


@router.post("/sessions")
def api_create_session(
    body: CreateSessionRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = create_session(
        db,
        user_id=user.id,
        title=body.title,
        kb_ids=body.kb_ids,
        model_id=body.model_id,
        system_prompt=body.system_prompt,
    )
    return {"id": session.id, "title": session.title}


@router.get("/sessions/{session_id}")
def api_get_session(
    session_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = get_session(db, session_id, user.id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "id": session.id,
        "title": session.title,
        "kb_ids": json.loads(session.knowledge_base_ids) if session.knowledge_base_ids else [],
        "model_id": session.model_id,
        "system_prompt": session.system_prompt,
    }


@router.put("/sessions/{session_id}")
def api_update_session(
    session_id: str,
    body: UpdateSessionRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    updates = body.model_dump(exclude_none=True)
    session = update_session(db, session_id, user.id, **updates)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"id": session.id, "title": session.title}


@router.delete("/sessions/{session_id}")
def api_delete_session(
    session_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ok = delete_session(db, session_id, user.id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}


# ── Messages ────────────────────────────────────────────────────────

@router.get("/sessions/{session_id}/messages")
def api_get_messages(
    session_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = get_session(db, session_id, user.id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = get_messages(db, session_id)

    # Build version counts for user messages that have edits
    from app.db.models import ChatMessage as CM
    version_counts = {}
    for m in messages:
        if m.role == "user":
            root_id = m.parent_id or m.id
            if root_id not in version_counts:
                count = (
                    db.query(CM)
                    .filter(
                        CM.session_id == session_id,
                        ((CM.parent_id == root_id) | (CM.id == root_id)),
                        CM.role == "user",
                    )
                    .count()
                )
                version_counts[root_id] = count

    result = []
    for m in messages:
        entry = {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "sources": json.loads(m.sources_json) if m.sources_json else None,
            "images": json.loads(m.images_json) if getattr(m, "images_json", None) else None,
            "parent_id": m.parent_id,
            "version": m.version,
            "is_active": m.is_active,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        if m.role == "user":
            root_id = m.parent_id or m.id
            entry["total_versions"] = version_counts.get(root_id, 1)
        result.append(entry)

    return result


@router.put("/sessions/{session_id}/messages/{message_id}/edit")
def api_edit_message(
    session_id: str,
    message_id: str,
    body: EditMessageRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Edit a user message — creates a new version and deactivates subsequent messages."""
    session = get_session(db, session_id, user.id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        new_msg = edit_user_message(db, session_id, message_id, body.content)
        return {
            "id": new_msg.id,
            "content": new_msg.content,
            "version": new_msg.version,
            "parent_id": new_msg.parent_id,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/sessions/{session_id}/messages/{message_id}/versions")
def api_get_message_versions(
    session_id: str,
    message_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get all versions of an edited message."""
    session = get_session(db, session_id, user.id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return get_message_versions(db, session_id, message_id)


@router.put("/sessions/{session_id}/messages/{message_id}/switch")
def api_switch_message_version(
    session_id: str,
    message_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Switch to an existing message version without creating a new one."""
    session = get_session(db, session_id, user.id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        msg = switch_to_version(db, session_id, message_id)
        return {
            "id": msg.id,
            "content": msg.content,
            "version": msg.version,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── Chat (RAG) ─────────────────────────────────────────────────────

@router.post("/sessions/{session_id}/chat", dependencies=[Depends(_require_llm_ready)])
@limiter.limit("10/minute")
def api_chat(
    request: Request,
    session_id: str,
    body: ChatRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Send a message and get a RAG-augmented response."""
    try:
        result = chat(
            db,
            session_id,
            user.id,
            body.message,
            images=body.images,
            message_id=body.message_id,
            metadata_filters=body.metadata_filters,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/sessions/{session_id}/chat/stream", dependencies=[Depends(_require_llm_ready)])
@limiter.limit("10/minute")
async def api_chat_stream(
    request: Request,
    session_id: str,
    body: ChatRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Streaming chat — returns Server-Sent Events."""
    try:
        return StreamingResponse(
            chat_stream(
                request,
                db,
                session_id,
                user.id,
                body.message,
                images=body.images,
                message_id=body.message_id,
                metadata_filters=body.metadata_filters,
            ),
            media_type="text/event-stream",
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

