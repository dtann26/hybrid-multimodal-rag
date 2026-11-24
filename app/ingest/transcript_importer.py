"""
Transcript importer — imports existing JSON transcript files into a
dynamically-specified knowledge base.

Reads the ``data/transcribe_result/*.json`` files produced by the
existing Whisper pipeline and creates Document records in the DB,
then indexes chunks directly into Milvus tagged with the target kb_id.

Key guarantees
--------------
* The target KB is validated in SQLite **before** any parsing/chunking.
* Milvus payloads are tagged with ``kb_id`` for filtered retrieval.
* On Milvus failure the SQLite document status is rolled back to
  ``"error"`` so the record stays coherent.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import get_project_root
from app.db.models import Folder, KnowledgeBase
from app.rag.chunker import chunk_transcript_segments
from app.services.document_service import (
    create_document,
    index_chunks_to_milvus,
    list_documents,
    update_document_status,
)
from app.models_manager.registry import encode_batch

logger = logging.getLogger(__name__)

TRANSCRIPTS_DIR = "data/transcribe_result"
YOUTUBE_DB_PATH = Path("data") / "youtube.db"


def get_transcript_filename_from_youtube_db(video_id: str) -> str | None:
    """Resolve transcript filename from youtube.db for a video id.

    Priority:
    1. videos.text_loc basename (if available)
    2. videos.title + ".json" (fallback)
    """
    db_path = get_project_root() / YOUTUBE_DB_PATH
    if not db_path.exists():
        logger.warning("youtube.db not found at %s; using local transcript filename", db_path)
        return None

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT text_loc, title FROM videos WHERE video_id = ? LIMIT 1",
            (video_id,),
        )
        row = cursor.fetchone()
    except sqlite3.Error as e:
        logger.warning("Failed to query youtube.db for video_id '%s': %s", video_id, e)
        return None
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass

    if not row:
        return None

    text_loc, title = row
    if title and str(title).strip():
        return str(title).strip()
    if text_loc and str(text_loc).strip():
        # Use ntpath to correctly extract basename from Windows-style paths
        # (e.g. D:\Videos\transcribe_result\abc.json) even when running on Linux.
        import ntpath
        return ntpath.basename(str(text_loc).strip())
    return None


def discover_transcript_files(base_dir: str | Path | None = None) -> list[Path]:
    """Find all .json transcript files (excluding .backup files)."""
    root = Path(base_dir) if base_dir else get_project_root() / TRANSCRIPTS_DIR
    if not root.exists():
        logger.warning("Transcripts directory not found: %s", root)
        return []
    files = sorted(root.glob("*.json"))
    files = [f for f in files if not f.name.endswith(".backup")]
    logger.info("Found %d transcript files in %s", len(files), root)
    return files


def resolve_kb(
    db: Session,
    *,
    kb_id: str | None = None,
    kb_name: str | None = None,
) -> KnowledgeBase:
    """Resolve and validate a KnowledgeBase by id or name.

    Raises ``ValueError`` if neither is provided or the KB is not found.
    """
    if kb_id is not None:
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
        if kb is None:
            raise ValueError(f"Knowledge base with id '{kb_id}' not found in the database")
        return kb

    if kb_name is not None:
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.name == kb_name).first()
        if kb is None:
            raise ValueError(f"Knowledge base with name '{kb_name}' not found in the database")
        return kb

    raise ValueError("Either kb_id or kb_name must be provided")


def import_transcript_file(
    db: Session,
    file_path: Path,
    folder_id: str,
    kb_id: str,
    collection_name: str,
    embedding_model_path: str | None = None,
    batch_size: int = 16,
) -> dict | None:
    """Import a single transcript JSON file into the specified knowledge base.

    Steps:
    1. Create a Document record in DB
    2. Parse and chunk the transcript segments
    3. Encode chunks with the embedding model
    4. Index into Milvus — payloads tagged with kb_id
    5. On Milvus failure: roll back SQLite document status to "error"
    """
    video_id = file_path.stem
    source_filename = get_transcript_filename_from_youtube_db(video_id) or file_path.name

    # Check if already imported (search existing docs in folder)
    existing_docs = list_documents(db, folder_id)
    doc_to_retry = None
    for doc in existing_docs:
        metadata = doc.get("metadata") or {}
        if (
            doc.get("title") in {file_path.name, source_filename}
            or (isinstance(metadata, dict) and metadata.get("video_id") == video_id)
        ):
            if doc.get("status") == "error":
                logger.info("Transcript '%s' previously failed, will retry", source_filename)
                doc_to_retry = doc
                break
            else:
                logger.info("Transcript '%s' already imported, skipping", source_filename)
                return doc

    # Read transcript
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error("Failed to read %s: %s", file_path, e)
        return None

    segments = data.get("segments", [])
    if not segments:
        logger.warning("No segments in %s", file_path.name)
        return None

    # Create document record in DB or reuse existing error doc
    if doc_to_retry:
        doc = doc_to_retry
    else:
        doc = create_document(
            db,
            folder_id=folder_id,
            title=source_filename,
            file_type="json",
            source_type="youtube",
            file_size=file_path.stat().st_size,
            source_url=f"https://www.youtube.com/watch?v={video_id}",
            metadata={
                "video_id": video_id,
                "segment_count": len(segments),
                "transcript_filename": source_filename,
            },
        )

    update_document_status(db, doc["id"], folder_id, "processing")

    # Chunk the transcript
    chunks = chunk_transcript_segments(segments)
    if not chunks:
        return update_document_status(db, doc["id"], folder_id, "error", error_message="No chunks generated") or doc

    # Encode chunks
    try:
        texts = [c.content for c in chunks]
        vectors = encode_batch(texts, model_path=embedding_model_path, batch_size=batch_size)
    except Exception as e:
        logger.error("Embedding failed for %s: %s", file_path.name, e)
        return update_document_status(db, doc["id"], folder_id, "error", error_message=f"Embedding failed: {e}") or doc

    # Index into Milvus — kb_id isolation is achieved at the collection level:
    # each KB has its own Milvus collection (kb.collection_name), so no extra
    # payload field is needed. The SQLite ChunkMetadata.document_id chain
    # (document → folder → knowledge_base) provides the structured link.
    try:
        index_chunks_to_milvus(
            db,
            doc_id=doc["id"],
            folder_id=folder_id,
            collection_name=collection_name,
            chunks=chunks,
            vectors=vectors,
            source_type="youtube",
        )
    except Exception as e:
        # Roll back SQLite status on Milvus failure to keep state coherent
        error_msg = f"Milvus indexing failed: {e}"
        logger.error("Milvus indexing failed for %s: %s", file_path.name, e)
        update_document_status(db, doc["id"], folder_id, "error", error_message=error_msg)
        return update_document_status(db, doc["id"], folder_id, "error", error_message=error_msg) or doc

    logger.info("Imported transcript '%s' — %d chunks", source_filename, len(chunks))
    return update_document_status(db, doc["id"], folder_id, "processed", chunk_count=len(chunks)) or doc


def import_all_transcripts(
    db: Session,
    kb_id: str | None = None,
    kb_name: str | None = None,
    folder_id: str | None = None,
    base_dir: str | Path | None = None,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Import all transcript files into the specified knowledge base.

    The KB can be identified by ``kb_id`` (preferred) or ``kb_name``.
    Validates the KB exists in SQLite before any file I/O begins.
    """
    # Validate & resolve KB — raises ValueError if not found
    kb = resolve_kb(db, kb_id=kb_id, kb_name=kb_name)
    kb_id = kb.id  # normalise to id for the rest of the function

    if folder_id is None:
        folder = db.query(Folder).filter(Folder.knowledge_base_id == kb_id).first()
        if folder is None:
            raise ValueError(f"No folders in KB {kb_id}")
        folder_id = folder.id

    embedding_path = None
    if kb.embedding_model_id:
        from app.models_manager.registry import get_model
        model = get_model(db, kb.embedding_model_id)
        if model:
            embedding_path = model.model_path

    files = discover_transcript_files(base_dir)
    imported = 0
    skipped = 0
    errors = 0

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from app.db.session import get_session_factory

    def process_file(f_path: Path) -> str:
        SessionLocal = get_session_factory()
        with SessionLocal() as local_db:
            try:
                # First check if doc is already skipped to avoid extra work
                existing_docs = list_documents(local_db, folder_id)
                video_id = f_path.stem
                source_filename = get_transcript_filename_from_youtube_db(video_id) or f_path.name
                for d in existing_docs:
                    metadata = d.get("metadata") or {}
                    if (
                        d.get("title") in {f_path.name, source_filename}
                        or (isinstance(metadata, dict) and metadata.get("video_id") == video_id)
                    ):
                        if d.get("status") == "error":
                            pass # allow it to be processed again
                        else:
                            return "skipped"

                doc = import_transcript_file(
                    db=local_db,
                    file_path=f_path,
                    folder_id=folder_id,
                    kb_id=kb.id,
                    collection_name=kb.collection_name,
                    embedding_model_path=embedding_path,
                    batch_size=batch_size,
                )
                
                if doc and doc.get("status") == "processed":
                    return "imported"
                elif doc and doc.get("status") == "error":
                    return "error"
                else:
                    return "skipped"
            except Exception as e:
                logger.error("Failed to import %s: %s", f_path.name, e)
                return "error"

    max_workers = 4 # use 4 threads

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, f): f for f in files}
        for future in as_completed(futures):
            f_path = futures[future]
            try:
                result = future.result()
                if result == "imported":
                    imported += 1
                elif result == "skipped":
                    skipped += 1
                else:
                    errors += 1
            except Exception as e:
                logger.error("Exception processing %s: %s", f_path.name, e)
                errors += 1

    summary = {
        "total_files": len(files),
        "imported": imported,
        "skipped": skipped,
        "errors": errors,
    }
    logger.info("Transcript import complete: %s", summary)
    return summary
