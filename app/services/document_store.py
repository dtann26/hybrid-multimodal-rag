"""
Document store — SQLAlchemy-backed document metadata storage.

All document metadata lives in the ``documents`` table.
Document chunks + vectors live in the Milvus vector database.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import Document

logger = logging.getLogger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _row_to_dict(doc: Document) -> dict[str, Any]:
    """Convert a Document ORM row to a plain dict (matches old JSON format)."""
    return {
        "id": doc.id,
        "folder_id": doc.folder_id,
        "title": doc.title,
        "file_type": doc.file_type,
        "source_type": doc.source_type,
        "file_size": doc.file_size,
        "status": doc.status,
        "error_message": doc.error_message,
        "chunk_count": doc.chunk_count,
        "source_url": doc.source_url,
        "metadata": json.loads(doc.metadata_json) if doc.metadata_json else {},
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
    }


# ── CRUD ────────────────────────────────────────────────────────────

def create_document(
    db: Session,
    folder_id: str,
    title: str,
    file_type: str,
    source_type: str = "upload",
    file_size: int = 0,
    source_url: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Create a new document record in the database.

    Returns the document as a dict.
    """
    doc = Document(
        id=str(uuid.uuid4()),
        folder_id=folder_id,
        title=title,
        file_type=file_type,
        source_type=source_type,
        file_size=file_size,
        source_url=source_url,
        metadata_json=json.dumps(metadata) if metadata else None,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    logger.info("Created document %s in folder %s", doc.id, folder_id)
    return _row_to_dict(doc)


def get_document(db: Session, doc_id: str, folder_id: str | None = None) -> dict | None:
    """Read a document record from the database."""
    q = db.query(Document).filter(Document.id == doc_id)
    if folder_id:
        q = q.filter(Document.folder_id == folder_id)
    row = q.first()
    return _row_to_dict(row) if row else None


def list_documents(db: Session, folder_id: str) -> list[dict]:
    """List all documents in a folder (sorted newest-first)."""
    rows = (
        db.query(Document)
        .filter(Document.folder_id == folder_id)
        .order_by(Document.created_at.desc())
        .all()
    )
    return [_row_to_dict(r) for r in rows]


def list_documents_by_kb(db: Session, folder_ids: list[str]) -> list[dict]:
    """List all documents across multiple folders (all folders in a KB)."""
    if not folder_ids:
        return []
    rows = (
        db.query(Document)
        .filter(Document.folder_id.in_(folder_ids))
        .order_by(Document.created_at.desc())
        .all()
    )
    return [_row_to_dict(r) for r in rows]


def update_document(db: Session, doc_id: str, folder_id: str, updates: dict) -> dict | None:
    """Update fields in a document record.  Returns the updated dict."""
    row = db.query(Document).filter(Document.id == doc_id, Document.folder_id == folder_id).first()
    if row is None:
        return None

    for key, value in updates.items():
        if key == "metadata":
            row.metadata_json = json.dumps(value) if value else None
        elif hasattr(row, key):
            setattr(row, key, value)

    row.updated_at = _utcnow()
    db.commit()
    db.refresh(row)
    return _row_to_dict(row)


def update_document_status(
    db: Session,
    doc_id: str,
    folder_id: str,
    status: str,
    error_message: str | None = None,
    chunk_count: int | None = None,
) -> dict | None:
    """Update the processing status of a document."""
    updates: dict[str, Any] = {"status": status}
    if error_message is not None:
        updates["error_message"] = error_message
    if chunk_count is not None:
        updates["chunk_count"] = chunk_count
    return update_document(db, doc_id, folder_id, updates)


def delete_document(db: Session, doc_id: str, folder_id: str | None = None) -> bool:
    """Delete a document record and its stored upload file."""
    q = db.query(Document).filter(Document.id == doc_id)
    if folder_id:
        q = q.filter(Document.folder_id == folder_id)
    row = q.first()
    if row is None:
        return False

    # Note: stored_filename was removed in schema consolidation.
    # File resolution now uses doc_id directly in document_service.py:
    # stored files are named as: {doc_id}{ext}
    # See app.services.document_service.save_uploaded_file()

    # Delete chunk JSON file if exists
    from app.services.chunk_store import delete_chunks
    delete_chunks(doc_id)

    db.delete(row)
    db.commit()
    logger.info("Deleted document %s", doc_id)
    return True


def delete_documents_in_folder(db: Session, folder_id: str) -> int:
    """Delete all documents in a folder.  Returns count deleted."""
    rows = db.query(Document).filter(Document.folder_id == folder_id).all()
    count = 0
    for row in rows:
        # Note: stored_filename was removed in schema consolidation.
        # File resolution now uses doc_id directly.

        from app.services.chunk_store import delete_chunks
        delete_chunks(row.id)

        db.delete(row)
        count += 1

    db.commit()
    return count


def count_documents(db: Session, folder_ids: list[str] | None = None) -> int:
    """Count total documents, optionally filtered by folder IDs."""
    q = db.query(func.count(Document.id))
    if folder_ids:
        q = q.filter(Document.folder_id.in_(folder_ids))
    return q.scalar() or 0


def count_documents_by_status(db: Session, folder_ids: list[str], status: str) -> int:
    """Count documents matching a particular status across folders."""
    return (
        db.query(func.count(Document.id))
        .filter(Document.folder_id.in_(folder_ids), Document.status == status)
        .scalar() or 0
    )


def get_all_documents(db: Session) -> list[dict]:
    """Get every document across all folders."""
    rows = db.query(Document).all()
    return [_row_to_dict(r) for r in rows]
