"""
Document service — file upload and Milvus indexing.

Document *metadata* is stored in the SQLite database (see document_store.py).
Document *chunks + vectors* are stored only in the Milvus vector database.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import get_storage_config
from app.db.models import ChunkMetadata, Document, Folder, KnowledgeBase
from app.rag.retriever import insert_vectors, delete_vectors_by_ids
from app.services.chunk_store import save_chunks as _save_chunks_json, delete_chunks as _delete_chunks_json
from app.services.document_store import (
    create_document,
    delete_document as _store_delete,
    get_document,
    list_documents,
    list_documents_by_kb,
    update_document,
    update_document_status,
    count_documents,
    count_documents_by_status,
    delete_documents_in_folder,
    get_all_documents,
)

logger = logging.getLogger(__name__)


def delete_document(db: Session, doc_id: str, folder_id: str | None = None) -> bool:
    """Delete a document, its chunk JSON, its FTS5 entries, and its Milvus vectors."""
    # 1. Collect chunk IDs and resolve collection name BEFORE deleting SQL rows
    chunk_ids = [
        row.id for row in
        db.query(ChunkMetadata.id).filter(ChunkMetadata.document_id == doc_id).all()
    ]

    collection_name: str | None = None
    doc_row = db.query(Document).filter(Document.id == doc_id).first()
    if doc_row and doc_row.folder_id:
        folder = db.query(Folder).filter(Folder.id == doc_row.folder_id).first()
        if folder:
            kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == folder.knowledge_base_id).first()
            if kb:
                collection_name = kb.collection_name

    # 2. Delete FTS5 entries
    if chunk_ids:
        from app.db.session import delete_from_fts
        try:
            delete_from_fts(db, chunk_ids)
            db.commit()
        except Exception as e:
            logger.warning("Failed to delete FTS5 entries for doc %s: %s", doc_id, e)
            db.rollback()

    # 3. Delete Milvus vectors (best-effort; log warning on failure)
    if collection_name:
        try:
            from app.rag.retriever import delete_by_document_id
            delete_by_document_id(collection_name, doc_id, db)
        except Exception as e:
            logger.warning(
                "Milvus vector deletion failed for doc %s (collection=%s): %s. "
                "Vectors may be orphaned until next collection rebuild.",
                doc_id, collection_name, e,
            )

    # 4. Delete chunk JSON file
    _delete_chunks_json(doc_id)

    # 5. Delete SQLite records (document + chunk metadata)
    return _store_delete(db, doc_id, folder_id)


# ── File upload handling ────────────────────────────────────────────

def save_uploaded_file(
    file_content: bytes,
    original_filename: str,
    doc_id: str,
) -> str:
    """Save uploaded file to disk, returns the doc_id (used as identifier).
    
    Files are stored as {doc_id}{ext} for consistent access.
    The original filename is now tracked in the 'title' field of the Document table.
    """
    storage = get_storage_config()
    upload_dir = Path(storage["upload_dir"])
    upload_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(original_filename).suffix
    stored_name = f"{doc_id}{ext}"
    dest = upload_dir / stored_name

    with open(dest, "wb") as f:
        f.write(file_content)

    return stored_name


# ── Milvus chunk indexing ──────────────────────────────────────────

def index_chunks_to_milvus(
    db: Session,
    doc_id: str,
    folder_id: str,
    collection_name: str,
    chunks: list[Any],
    vectors: list[list[float]],
    source_type: str = "upload",
) -> None:
    """Insert chunk vectors into Milvus and update the document status.

    *chunks* is a list of ``Chunk`` dataclass instances (from app.rag.chunker)
    that carry ``content``, ``start_time``, ``end_time``, ``page_start``,
    ``page_end``, and ``chunk_index``.
    """
    n = len(chunks)
    chunk_ids = [str(uuid.uuid4()) for _ in range(n)]

    # Save chunk metadata to SQLite

    # Clear any previously processed chunks for this document from both FTS5 and SQLite
    old_chunk_ids = [
        row.id for row in
        db.query(ChunkMetadata.id).filter(ChunkMetadata.document_id == doc_id).all()
    ]
    if old_chunk_ids:
        from app.db.session import delete_from_fts
        try:
            delete_from_fts(db, old_chunk_ids)
        except Exception as e:
            logger.warning("Failed to delete old FTS5 entries for doc %s: %s", doc_id, e)

    db.query(ChunkMetadata).filter(ChunkMetadata.document_id == doc_id).delete()

    for i, c in enumerate(chunks):
        chunk_obj = ChunkMetadata(
            id=chunk_ids[i],
            document_id=doc_id,
            chunk_index=c.index,
            content=c.content,
            source_type=source_type,
            start_time=c.start_time or 0.0,
            end_time=c.end_time or 0.0,
            page_start=c.page_start,
            page_end=c.page_end,
        )
        db.add(chunk_obj)

    # R5: Insert into FTS5 index atomically in the same transaction
    from app.db.session import insert_batch_into_fts
    fts_items = [(chunk_ids[i], c.content) for i, c in enumerate(chunks)]
    insert_batch_into_fts(db, fts_items)

    db.commit()

    # Save chunk data to JSON file (data/chunks/{doc_id}.json)
    _save_chunks_json(
        doc_id=doc_id,
        folder_id=folder_id,
        chunks=chunks,
        source_type=source_type,
        chunk_ids=chunk_ids,
    )

    insert_vectors(
        collection_name=collection_name,
        chunk_ids=chunk_ids,
        vectors=vectors,
    )

    # Update document status in DB
    update_document_status(db, doc_id, folder_id, "processed", chunk_count=n)
    logger.info("Indexed %d chunks for document %s into %s", n, doc_id, collection_name)
