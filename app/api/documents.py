"""
Documents API — file upload, list documents, trigger processing.

Document metadata is stored in the SQLite database.
Chunks are saved to JSON. Vector indexing is a separate step.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_admin, get_db
from app.db.models import Folder, KnowledgeBase, User
from app.ingest.processor import SUPPORTED_TYPES
from app.services.document_service import (
    create_document,
    delete_document,
    get_document,
    list_documents,
    list_documents_by_kb,
    save_uploaded_file,
    update_document,
    update_document_status,
)

router = APIRouter()


@router.get("/kb/{kb_id}")
def api_list_documents_by_kb(
    kb_id: str,
    page: int = Query(1, ge=1),
    limit: int = Query(100, ge=1, le=1000),
    search: str = Query("", description="Search by filename"),
    status: str = Query("", description="Filter by status"),
    source_type: str = Query("", description="Filter by source type"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List documents in a KB with pagination and search."""
    folders = db.query(Folder).filter(Folder.knowledge_base_id == kb_id).all()
    folder_ids = [f.id for f in folders]
    all_docs = list_documents_by_kb(db, folder_ids)

    # Filter by search query
    if search:
        q = search.lower()
        all_docs = [d for d in all_docs if q in d.get("title", "").lower()]

    # Filter by status
    if status:
        all_docs = [d for d in all_docs if d.get("status") == status]

    # Filter by source type
    if source_type:
        all_docs = [d for d in all_docs if d.get("source_type") == source_type]

    total = len(all_docs)
    total_pages = max(1, (total + limit - 1) // limit)

    # Paginate
    start = (page - 1) * limit
    page_docs = all_docs[start : start + limit]

    return {
        "documents": [
            {
                "id": d["id"],
                "folder_id": d.get("folder_id"),
                "title": d["title"],
                "file_type": d["file_type"],
                "source_type": d["source_type"],
                "file_size": d["file_size"],
                "status": d["status"],
                "chunk_count": d["chunk_count"],
                "error_message": d.get("error_message"),
                "created_at": d.get("created_at"),
            }
            for d in page_docs
        ],
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": total_pages,
    }


@router.get("/folder/{folder_id}")
def api_list_documents_by_folder(
    folder_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    docs = list_documents(db, folder_id)
    return [
        {
            "id": d["id"],
            "title": d["title"],
            "file_type": d["file_type"],
            "status": d["status"],
            "chunk_count": d["chunk_count"],
            "created_at": d.get("created_at"),
        }
        for d in docs
    ]


@router.post("/upload/{folder_id}", status_code=201)
async def api_upload_documents(
    folder_id: str,
    files: List[UploadFile] = File(...),
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Upload one or more document files.

    Files are saved to disk and chunked (text extraction + chunking).
    Vector indexing is NOT performed — use the ``/index`` endpoint after.

    Returns a list of results per file, showing which succeeded, which
    were incompatible, and any errors.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Verify folder exists
    folder = db.query(Folder).filter(Folder.id == folder_id).first()
    if folder is None:
        raise HTTPException(status_code=404, detail="Folder not found")

    results = []

    for file in files:
        filename = file.filename or "unnamed"
        ext = Path(filename).suffix.lstrip(".").lower()

        # Check file type compatibility
        if ext not in SUPPORTED_TYPES:
            results.append({
                "filename": filename,
                "status": "incompatible",
                "error": f"Unsupported file type: .{ext}. Supported: {', '.join(sorted(SUPPORTED_TYPES))}",
                "id": None,
            })
            continue

        try:
            content = await file.read()

            if len(content) == 0:
                results.append({
                    "filename": filename,
                    "status": "error",
                    "error": "File is empty",
                    "id": None,
                })
                continue

            # Create document record in DB
            doc = create_document(
                db,
                folder_id=folder_id,
                title=filename,
                file_type=ext,
                source_type="upload",
                file_size=len(content),
            )

            # Save file to disk
            stored_name = save_uploaded_file(content, filename, doc["id"])

            # Chunk the document (no embedding/Milvus)
            from app.ingest.processor import chunk_document
            try:
                ok = chunk_document(db, doc["id"], folder_id=folder_id)
            except Exception as e:
                update_document_status(db, doc["id"], folder_id, "error", error_message=str(e))
                ok = False

            # Re-read updated doc
            doc = get_document(db, doc["id"], folder_id) or doc
            results.append({
                "filename": doc["title"],
                "status": doc.get("status", "error"),
                "chunk_count": doc.get("chunk_count", 0),
                "id": doc["id"],
                "error": doc.get("error_message"),
            })

        except Exception as e:
            results.append({
                "filename": filename,
                "status": "error",
                "error": str(e),
                "id": None,
            })

    return {
        "uploaded": len([r for r in results if r["status"] not in ("incompatible", "error")]),
        "incompatible": len([r for r in results if r["status"] == "incompatible"]),
        "errors": len([r for r in results if r["status"] == "error"]),
        "results": results,
    }


class IndexRequest(BaseModel):
    doc_ids: list[str]


class DeleteBatchRequest(BaseModel):
    doc_ids: list[str]


@router.post("/delete-batch")
def api_delete_documents_batch(
    body: DeleteBatchRequest,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Delete multiple documents by ID."""
    deleted = 0
    errors = 0
    for doc_id in body.doc_ids:
        try:
            ok = delete_document(db, doc_id)
            if ok:
                deleted += 1
            else:
                errors += 1
        except Exception:
            errors += 1
    return {"deleted": deleted, "errors": errors}


@router.post("/index/{folder_id}")
def api_index_documents(
    folder_id: str,
    body: IndexRequest,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Embed and index chunked documents into Milvus.

    Only documents with status ``"chunked"`` will be processed.
    """
    from app.ingest.processor import index_document

    results = []
    for doc_id in body.doc_ids:
        doc = get_document(db, doc_id, folder_id)
        if doc is None:
            results.append({"id": doc_id, "status": "not_found", "error": "Document not found"})
            continue
        if doc.get("status") not in ("chunked", "error", "pending"):
            results.append({
                "id": doc_id,
                "status": doc.get("status"),
                "error": f"Document status is '{doc.get('status')}', expected 'chunked'",
            })
            continue

        try:
            ok = index_document(db, doc_id, folder_id=folder_id)
            doc = get_document(db, doc_id, folder_id) or doc
            results.append({
                "id": doc_id,
                "status": doc.get("status", "error"),
                "chunk_count": doc.get("chunk_count", 0),
            })
        except Exception as e:
            results.append({"id": doc_id, "status": "error", "error": str(e)})

    indexed = len([r for r in results if r["status"] == "processed"])
    return {
        "indexed": indexed,
        "errors": len(results) - indexed,
        "results": results,
    }


@router.get("/{doc_id}")
def api_get_document(
    doc_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    doc = get_document(db, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "id": doc["id"],
        "title": doc["title"],
        "file_type": doc["file_type"],
        "source_type": doc["source_type"],
        "file_size": doc["file_size"],
        "status": doc["status"],
        "chunk_count": doc["chunk_count"],
        "error_message": doc.get("error_message"),
        "source_url": doc.get("source_url"),
        "metadata": doc.get("metadata"),
        "created_at": doc.get("created_at"),
    }


@router.delete("/{doc_id}")
def api_delete_document(
    doc_id: str,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    ok = delete_document(db, doc_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"ok": True}
