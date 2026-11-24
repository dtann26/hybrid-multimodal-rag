"""
Chat service — session & message management, RAG query orchestration.
"""

from __future__ import annotations

import json
import logging
import asyncio
from collections.abc import Generator
from typing import Any, AsyncGenerator

from fastapi import Request
from sqlalchemy.orm import Session

from app.core.config import get_rag_config
from app.db.models import ChatMessage, ChatSession
from app.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

ACTIVE_STREAMS = {}


from app.services.memory_service import MemoryManager
from app.rag.planner import plan_rag_queries


# ── Session CRUD ────────────────────────────────────────────────────

def create_session(
    db: Session,
    user_id: str,
    title: str = "New Chat",
    kb_ids: list[str] | None = None,
    model_id: str | None = None,
    system_prompt: str | None = None,
) -> ChatSession:
    session = ChatSession(
        user_id=user_id,
        title=title,
        knowledge_base_ids=json.dumps(kb_ids or []),
        model_id=model_id,
        system_prompt=system_prompt,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_session(db: Session, session_id: str, user_id: str) -> ChatSession | None:
    return (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
        .first()
    )


def list_sessions(db: Session, user_id: str) -> list[ChatSession]:
    return (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user_id)
        .order_by(ChatSession.updated_at.desc())
        .all()
    )


def update_session(
    db: Session,
    session_id: str,
    user_id: str,
    **kwargs,
) -> ChatSession | None:
    session = get_session(db, session_id, user_id)
    if session is None:
        return None
    for key, val in kwargs.items():
        if key == "kb_ids":
            session.knowledge_base_ids = json.dumps(val)
        elif hasattr(session, key):
            setattr(session, key, val)
    db.commit()
    db.refresh(session)
    return session


def delete_session(db: Session, session_id: str, user_id: str) -> bool:
    session = get_session(db, session_id, user_id)
    if session is None:
        return False
        
    task = ACTIVE_STREAMS.get(session_id)
    if task and not task.done():
        logger.warning(f"Cancelling active stream task for session {session_id} before deletion.")
        try:
            task.get_loop().call_soon_threadsafe(task.cancel)
        except Exception as e:
            logger.error(f"Failed to cancel task: {e}")
            
    db.delete(session)
    db.commit()
    return True


# ── Messages ────────────────────────────────────────────────────────

def get_messages(db: Session, session_id: str) -> list[ChatMessage]:
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id, ChatMessage.is_active == True)
        .order_by(ChatMessage.created_at)
        .all()
    )
    
    # Calculate and cache token_count for any messages where it is missing
    dirty = False
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        encoding = None

    for m in messages:
        if m.token_count is None:
            if encoding:
                m.token_count = len(encoding.encode(m.content or ""))
            else:
                m.token_count = len(m.content or "") // 4
            dirty = True
            
    if dirty:
        try:
            db.commit()
        except Exception:
            db.rollback()
            
    return messages


def add_message(
    db: Session,
    session_id: str,
    role: str,
    content: str,
    sources: list[dict] | None = None,
    images: list[str] | None = None,
) -> ChatMessage:
    msg = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        sources_json=json.dumps(sources) if sources else None,
        images_json=json.dumps(images) if images else None,
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


def edit_user_message(
    db: Session,
    session_id: str,
    message_id: str,
    new_content: str,
) -> ChatMessage:
    """Create a new version of a user message.

    - The original message (or its parent) becomes the parent_id.
    - All subsequent messages after the edited one are deactivated.
    - Returns the new active user message.
    """
    original = db.query(ChatMessage).filter(ChatMessage.id == message_id).first()
    if original is None:
        raise ValueError("Message not found")

    # Determine the root parent (the very first version)
    root_id = original.parent_id or original.id

    # Count existing versions to get the next version number
    max_version = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.session_id == session_id,
            ((ChatMessage.parent_id == root_id) | (ChatMessage.id == root_id)),
            ChatMessage.role == "user",
        )
        .count()
    )

    # Deactivate the current active user message in this group
    original.is_active = False

    # Deactivate all messages that come AFTER this one in the conversation
    later_messages = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.session_id == session_id,
            ChatMessage.created_at > original.created_at,
            ChatMessage.is_active == True,
        )
        .all()
    )
    for m in later_messages:
        m.is_active = False

    # Create the new version
    new_msg = ChatMessage(
        session_id=session_id,
        role="user",
        content=new_content,
        parent_id=root_id,
        version=max_version + 1,
        is_active=True,
    )
    db.add(new_msg)
    db.commit()
    db.refresh(new_msg)
    return new_msg


def get_message_versions(
    db: Session,
    session_id: str,
    message_id: str,
) -> list[dict]:
    """Return all versions of a message (original + edits)."""
    msg = db.query(ChatMessage).filter(ChatMessage.id == message_id).first()
    if msg is None:
        return []

    root_id = msg.parent_id or msg.id

    versions = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.session_id == session_id,
            ((ChatMessage.parent_id == root_id) | (ChatMessage.id == root_id)),
            ChatMessage.role == "user",
        )
        .order_by(ChatMessage.version)
        .all()
    )
    return [
        {
            "id": v.id,
            "content": v.content,
            "version": v.version,
            "is_active": v.is_active,
            "created_at": v.created_at.isoformat() if v.created_at else None,
        }
        for v in versions
    ]


def switch_to_version(
    db: Session,
    session_id: str,
    message_id: str,
) -> ChatMessage:
    """Switch the active version to a specific message without creating a new one.

    Deactivates all sibling versions, then activates the target version
    AND all paired assistant responses and descendant user prompts on its branch.
    """
    target = db.query(ChatMessage).filter(ChatMessage.id == message_id).first()
    if target is None:
        raise ValueError("Message not found")

    root_id = target.parent_id or target.id

    # Get ALL sibling user versions sorted by created_at
    siblings = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.session_id == session_id,
            ((ChatMessage.parent_id == root_id) | (ChatMessage.id == root_id)),
            ChatMessage.role == "user",
        )
        .order_by(ChatMessage.created_at)
        .all()
    )

    # Find the earliest sibling to know where the edit group starts
    earliest = siblings[0] if siblings else target

    # Deactivate ALL messages from the earliest sibling onward
    all_after = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.session_id == session_id,
            ChatMessage.created_at >= earliest.created_at,
        )
        .all()
    )
    for m in all_after:
        m.is_active = False

    from datetime import datetime

    def activate_branch_from(user_msg: ChatMessage, end_time: datetime | None):
        user_msg.is_active = True
        
        # Find siblings of this user message to know when its next sibling occurs
        u_root_id = user_msg.parent_id or user_msg.id
        u_siblings = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.session_id == session_id,
                ((ChatMessage.parent_id == u_root_id) | (ChatMessage.id == u_root_id)),
                ChatMessage.role == "user",
            )
            .order_by(ChatMessage.created_at)
            .all()
        )
        
        # Find next sibling
        u_next_sibling = None
        for idx, sib in enumerate(u_siblings):
            if sib.id == user_msg.id and idx + 1 < len(u_siblings):
                u_next_sibling = u_siblings[idx + 1]
                break
                
        branch_end = u_next_sibling.created_at if u_next_sibling else None
        if end_time:
            if branch_end is None or branch_end > end_time:
                branch_end = end_time
                
        # Find paired assistant message
        assistant_query = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.session_id == session_id,
                ChatMessage.role == "assistant",
                ChatMessage.created_at > user_msg.created_at,
            )
        )
        if branch_end:
            assistant_query = assistant_query.filter(ChatMessage.created_at < branch_end)
            
        paired_assistant = assistant_query.order_by(ChatMessage.created_at).first()
        if paired_assistant:
            paired_assistant.is_active = True
            
            # Find any next user message group after this assistant but before branch_end
            next_user_query = (
                db.query(ChatMessage)
                .filter(
                    ChatMessage.session_id == session_id,
                    ChatMessage.role == "user",
                    ChatMessage.created_at > paired_assistant.created_at,
                )
            )
            if branch_end:
                next_user_query = next_user_query.filter(ChatMessage.created_at < branch_end)
                
            next_user_msgs = next_user_query.order_by(ChatMessage.created_at).all()
            if next_user_msgs:
                first_next_user = next_user_msgs[0]
                nu_root_id = first_next_user.parent_id or first_next_user.id
                nu_group = [m for m in next_user_msgs if (m.parent_id == nu_root_id or m.id == nu_root_id)]
                latest_nu = nu_group[-1]
                activate_branch_from(latest_nu, branch_end)

    # Find the next sibling of our target user message to define the end of our target window
    target_next_sibling = None
    for idx, sib in enumerate(siblings):
        if sib.id == target.id and idx + 1 < len(siblings):
            target_next_sibling = siblings[idx + 1]
            break
            
    target_end_time = target_next_sibling.created_at if target_next_sibling else None
    
    # Activate target and its branch descendants
    activate_branch_from(target, target_end_time)

    # Also reactivate messages before the edit group
    before_group = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.session_id == session_id,
            ChatMessage.created_at < earliest.created_at,
        )
        .all()
    )
    for m in before_group:
        m.is_active = True

    db.commit()
    db.refresh(target)
    return target


# ── Chat with RAG ──────────────────────────────────────────────────

def chat(
    db: Session,
    session_id: str,
    user_id: str,
    user_message: str,
    images: list[str] | None = None,
    message_id: str | None = None,
    metadata_filters: dict | None = None,
) -> dict[str, Any]:
    """Send a user message and get a RAG-augmented response (blocking)."""
    session = get_session(db, session_id, user_id)
    if session is None:
        raise ValueError("Session not found")

    if message_id:
        # Deactivate all active messages that come AFTER this user message
        user_msg = db.query(ChatMessage).filter(ChatMessage.id == message_id).first()
        if user_msg:
            later_messages = (
                db.query(ChatMessage)
                .filter(
                    ChatMessage.session_id == session_id,
                    ChatMessage.created_at > user_msg.created_at,
                    ChatMessage.is_active == True,
                )
                .all()
            )
            for m in later_messages:
                m.is_active = False
            db.commit()
    else:
        # Save user message
        add_message(db, session_id, "user", user_message, images=images)

    # Parse KB ids
    kb_ids = json.loads(session.knowledge_base_ids) if session.knowledge_base_ids else []

    # Build chat history
    history = [
        {"role": m.role, "content": m.content, "token_count": m.token_count}
        for m in get_messages(db, session_id)[:-1]  # exclude the message we just added
    ]

    # Agentic Context Orchestration
    memory_manager = MemoryManager()
    user_memories = memory_manager.get_relevant_memories(db, user_id, user_message)

    plan = plan_rag_queries(user_message, history, user_memories)

    # Run RAG pipeline
    pipeline = RAGPipeline(db)
    answer, sources = pipeline.query(
        user_message=user_message,
        plan=plan,
        kb_ids=kb_ids,
        system_prompt=session.system_prompt,
        chat_history=history,
        user_memories=user_memories,
        model_id=session.model_id,
        images=images,
        metadata_filters=metadata_filters,
    )

    # Save assistant message
    assistant_msg = add_message(db, session_id, "assistant", answer, sources)

    # Auto-title on first exchange
    if len(history) <= 1:
        session.title = user_message[:100]
        db.commit()

    return {
        "message_id": assistant_msg.id,
        "content": answer,
        "sources": sources,
    }



async def chat_stream(
    request: Request,
    db: Session,
    session_id: str,
    user_id: str,
    user_message: str,
    images: list[str] | None = None,
    message_id: str | None = None,
    metadata_filters: dict | None = None,
) -> AsyncGenerator[str, None]:
    """Streaming variant — yields SSE-formatted tokens."""
    task = asyncio.current_task()
    if task:
        ACTIVE_STREAMS[session_id] = task

    try:
        session = get_session(db, session_id, user_id)
        if session is None:
            raise ValueError("Session not found")

        if message_id:
            # Deactivate all active messages that come AFTER this user message
            user_msg = db.query(ChatMessage).filter(ChatMessage.id == message_id).first()
            if user_msg:
                later_messages = (
                    db.query(ChatMessage)
                    .filter(
                        ChatMessage.session_id == session_id,
                        ChatMessage.created_at > user_msg.created_at,
                        ChatMessage.is_active == True,
                    )
                    .all()
                )
                for m in later_messages:
                    m.is_active = False
                db.commit()
        else:
            # Save user message
            add_message(db, session_id, "user", user_message, images=images)

        kb_ids = json.loads(session.knowledge_base_ids) if session.knowledge_base_ids else []

        history = [
            {"role": m.role, "content": m.content, "token_count": m.token_count}
            for m in get_messages(db, session_id)[:-1]
        ]

        # Agentic Context Orchestration
        memory_manager = MemoryManager()
        user_memories = memory_manager.get_relevant_memories(db, user_id, user_message)

        plan = plan_rag_queries(user_message, history, user_memories)

        pipeline = RAGPipeline(db)
        full_response = []
        sources = []

        try:
            gen = pipeline.query_stream(
                user_message=user_message,
                plan=plan,
                kb_ids=kb_ids,
                system_prompt=session.system_prompt,
                chat_history=history,
                user_memories=user_memories,
                model_id=session.model_id,
                images=images,
                metadata_filters=metadata_filters,
            )

            def _safe_next(g):
                try:
                    return next(g)
                except StopIteration as exc:
                    return StopIteration(exc.value)

            while True:
                if await request.is_disconnected():
                    logger.warning(f"Client disconnected. Halting stream for session {session_id}.")
                    try:
                        gen.close()
                    except ValueError:
                        pass
                    return

                res = await asyncio.to_thread(_safe_next, gen)
                if isinstance(res, StopIteration):
                    sources = res.value or []
                    break

                token = res
                full_response.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"
        except asyncio.CancelledError:
            logger.warning(f"Stream cancelled for session {session_id}. Halting vLLM.")
            if 'gen' in locals():
                try:
                    gen.close()
                except ValueError:
                    pass
            return
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("Chat stream error: %s", e)
            error_msg = f"Error: {type(e).__name__}: {e}"
            yield f"data: {json.dumps({'token': error_msg})}\n\n"
            full_response.append(error_msg)

        # Save the complete response
        answer = "".join(full_response)
        
        try:
            # Re-fetch session to ensure it wasn't deleted mid-stream
            current_session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            if current_session:
                add_message(db, session_id, "assistant", answer, sources)
                # Auto-title
                if len(history) <= 1:
                    current_session.title = user_message[:100]
                    db.commit()
            else:
                logger.warning(f"Session {session_id} deleted mid-stream. Skipping DB save.")
        except Exception as e:
            logger.error(f"Error saving stream message: {e}")
            db.rollback()

        yield f"data: {json.dumps({'done': True, 'sources': sources})}\n\n"
    finally:
        if ACTIVE_STREAMS.get(session_id) == task:
            del ACTIVE_STREAMS[session_id]
