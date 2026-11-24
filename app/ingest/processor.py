"""
Document processor — orchestrates file conversion, chunking, embedding,
and indexing for uploaded documents.

Document metadata lives in the SQLite database (document_store).
Chunks + vectors go directly into Milvus — nothing is duplicated in the
SQL database.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import get_storage_config
from app.db.models import Folder, KnowledgeBase
from app.ingest.markdown_converter import convert_to_markdown
from app.ingest.pdf_extractor import extract_pdf_images
from app.rag.chunker import chunk_text, chunk_pages
from app.services.document_service import (
    get_document,
    index_chunks_to_milvus,
    update_document_status,
    update_document,
)
from app.services.chunk_store import save_chunks as _save_chunks_json
from app.models_manager.registry import encode_batch

logger = logging.getLogger(__name__)

# File types that MarkItDown can convert
SUPPORTED_TYPES = {
    "pdf", "docx", "pptx", "xlsx", "html", "htm",
    "txt", "md", "csv", "json", "xml", "rtf",
}


def _extract_pdf_pages(file_path: Path) -> list[dict]:
    """Extract text per page from a PDF using PyMuPDF.

    Returns a list of ``{"page": 1, "text": "..."}`` dicts.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF not installed — falling back to plain text chunking for PDF")
        return []

    pages: list[dict] = []
    doc = fitz.open(file_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        if text.strip():
            pages.append({"page": page_num + 1, "text": text})
    doc.close()
    return pages


def chunk_document(
    db: Session,
    doc_id: str,
    folder_id: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> bool:
    """Chunk an uploaded document and save chunks to JSON — NO embedding or Milvus indexing.

    1. Convert to text (page-by-page for PDFs, Markdown for others)
    2. Chunk the text (with page tracking for PDFs)
    3. Save chunk data to ``data/chunks/{doc_id}.json``

    Returns True on success.  Status will be set to ``"chunked"``.
    """
    doc = get_document(db, doc_id, folder_id)
    if doc is None:
        logger.error("Document %s not found", doc_id)
        return False

    fid = doc["folder_id"]
    update_document_status(db, doc_id, fid, "processing")

    # Resolve paths
    # Files are stored as {doc_id}{ext} in the upload directory
    # The file extension is determined from the original file_type
    storage = get_storage_config()
    file_type = doc.get("file_type", "").lower()
    ext = f".{file_type}" if file_type else ""
    stored_filename = f"{doc_id}{ext}"
    file_path = Path(storage["upload_dir"]) / stored_filename
    if file_path is None or not file_path.exists():
        update_document_status(db, doc_id, fid, "error", error_message="File not found on disk")
        return False

    try:

        if file_type == "pdf":
            pages = _extract_pdf_pages(file_path)
            if pages:
                chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            else:
                text = convert_to_markdown(file_path)
                if not text.strip():
                    update_document_status(db, doc_id, fid, "error", error_message="No text extracted")
                    return False
                chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            # Extract images (best effort)
            images_dir = Path(storage["upload_dir"]) / "images" / doc_id
            try:
                extract_pdf_images(file_path, images_dir)
            except Exception as e:
                logger.warning("Image extraction failed for %s: %s", doc_id, e)
        else:
            text = convert_to_markdown(file_path)
            if not text.strip():
                update_document_status(db, doc_id, fid, "error", error_message="No text extracted")
                return False
            chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        if not chunks:
            update_document_status(db, doc_id, fid, "error", error_message="No chunks generated")
            return False

        # Save chunk data to JSON file
        _save_chunks_json(
            doc_id=doc_id,
            folder_id=fid,
            chunks=chunks,
            source_type=doc.get("source_type", "upload"),
        )

        update_document_status(db, doc_id, fid, "chunked", chunk_count=len(chunks))
        logger.info("Chunked document '%s' — %d chunks (not indexed)", doc.get("title"), len(chunks))
        return True

    except Exception as e:
        update_document_status(db, doc_id, fid, "error", error_message=str(e))
        logger.exception("Failed to chunk document %s", doc_id)
        return False


def index_document(
    db: Session,
    doc_id: str,
    folder_id: str | None = None,
) -> bool:
    """Embed and index an already-chunked document into Milvus.

    Reads chunks from the JSON file at ``data/chunks/{doc_id}.json``,
    encodes them, and inserts into the KB's Milvus collection.

    Returns True on success.  Status will be set to ``"processed"``.
    """
    from app.services.chunk_store import load_chunks
    from app.rag.chunker import Chunk

    doc = get_document(db, doc_id, folder_id)
    if doc is None:
        logger.error("Document %s not found", doc_id)
        return False

    fid = doc["folder_id"]

    # Load chunks from JSON
    chunk_data = load_chunks(doc_id)
    if chunk_data is None or not chunk_data.get("chunks"):
        update_document_status(db, doc_id, fid, "error", error_message="No chunk data found — chunk the document first")
        return False

    # Get KB info
    folder = db.query(Folder).filter(Folder.id == fid).first()
    if folder is None:
        update_document_status(db, doc_id, fid, "error", error_message="Folder not found")
        return False

    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == folder.knowledge_base_id).first()
    if kb is None or not kb.collection_name:
        update_document_status(db, doc_id, fid, "error", error_message="Knowledge base not configured")
        return False

    try:
        update_document_status(db, doc_id, fid, "indexing")

        # Reconstruct Chunk objects from JSON
        raw_chunks = chunk_data["chunks"]
        chunks = []
        for rc in raw_chunks:
            chunks.append(Chunk(
                index=rc.get("chunk_index", 0),
                content=rc["content"],
                start_time=rc.get("start_time"),
                end_time=rc.get("end_time"),
                page_start=rc.get("page_start", 0),
                page_end=rc.get("page_end", 0),
                metadata=rc.get("metadata", {}),
            ))

        # Encode
        texts = [c.content for c in chunks]
        embedding_path = None
        if kb.embedding_model_id:
            from app.models_manager.registry import get_model
            model = get_model(db, kb.embedding_model_id)
            if model:
                embedding_path = model.model_path

        vectors = encode_batch(texts, model_path=embedding_path)

        # Index into Milvus
        index_chunks_to_milvus(
            db,
            doc_id=doc_id,
            folder_id=fid,
            collection_name=kb.collection_name,
            chunks=chunks,
            vectors=vectors,
            source_type=doc.get("source_type", "upload"),
        )

        logger.info("Indexed document '%s' — %d vectors", doc.get("title"), len(chunks))
        return True

    except Exception as e:
        update_document_status(db, doc_id, fid, "error", error_message=str(e))
        logger.exception("Failed to index document %s", doc_id)
        return False


def process_document(
    db: Session,
    doc_id: str,
    folder_id: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> bool:
    """Process an uploaded document end-to-end.

    1. Convert to text (page-by-page for PDFs, Markdown for others)
    2. Chunk the text (with page tracking for PDFs)
    3. Encode chunks with embedding model
    4. Index into Milvus

    Returns True on success.
    """
    doc = get_document(db, doc_id, folder_id)
    if doc is None:
        logger.error("Document %s not found", doc_id)
        return False

    fid = doc["folder_id"]
    update_document_status(db, doc_id, fid, "processing")

    # Resolve paths
    # Files are stored as {doc_id}{ext} in the upload directory
    storage = get_storage_config()
    file_type = doc.get("file_type", "").lower()
    ext = f".{file_type}" if file_type else ""
    stored_filename = f"{doc_id}{ext}"
    file_path = Path(storage["upload_dir"]) / stored_filename
    if file_path is None or not file_path.exists():
        update_document_status(db, doc_id, fid, "error", error_message="File not found on disk")
        return False

    # Get KB info
    folder = db.query(Folder).filter(Folder.id == fid).first()
    if folder is None:
        update_document_status(db, doc_id, fid, "error", error_message="Folder not found")
        return False

    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == folder.knowledge_base_id).first()
    if kb is None or not kb.collection_name:
        update_document_status(db, doc_id, fid, "error", error_message="Knowledge base not configured")
        return False

    try:
        file_type = doc.get("file_type", "").lower()

        # ── 1. Choose chunking strategy ──────────────────────────────
        if file_type == "pdf":
            # Try page-aware extraction first
            pages = _extract_pdf_pages(file_path)
            if pages:
                chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            else:
                # Fallback to plain text
                text = convert_to_markdown(file_path)
                if not text.strip():
                    update_document_status(db, doc_id, fid, "error", error_message="No text extracted")
                    return False
                chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            # Extract images (best effort)
            images_dir = Path(storage["upload_dir"]) / "images" / doc_id
            try:
                extract_pdf_images(file_path, images_dir)
            except Exception as e:
                logger.warning("Image extraction failed for %s: %s", doc_id, e)
        else:
            # Non-PDF: convert to markdown, then plain chunk
            text = convert_to_markdown(file_path)
            if not text.strip():
                update_document_status(db, doc_id, fid, "error", error_message="No text extracted")
                return False
            chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        if not chunks:
            update_document_status(db, doc_id, fid, "error", error_message="No chunks generated")
            return False

        # ── 2. Encode chunks ─────────────────────────────────────────
        texts = [c.content for c in chunks]
        embedding_path = None
        if kb.embedding_model_id:
            from app.models_manager.registry import get_model
            model = get_model(db, kb.embedding_model_id)
            if model:
                embedding_path = model.model_path

        vectors = encode_batch(texts, model_path=embedding_path)

        # ── 3. Index into Milvus ─────────────────────────────────────
        index_chunks_to_milvus(
            db,
            doc_id=doc_id,
            folder_id=fid,
            collection_name=kb.collection_name,
            chunks=chunks,
            vectors=vectors,
            source_type=doc.get("source_type", "upload"),
        )

        logger.info("Processed document '%s' — %d chunks", doc.get("title"), len(chunks))
        return True

    except Exception as e:
        update_document_status(db, doc_id, fid, "error", error_message=str(e))
        logger.exception("Failed to process document %s", doc_id)
        return False
