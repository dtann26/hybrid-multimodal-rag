"""
Chunk store — JSON file-based chunk data storage.

Chunk results from document processing are saved as JSON files on disk
instead of only living in the Milvus vector database.  This provides:

- A cheap, persistent backup of all chunk data and metadata
- The ability to rebuild Milvus collections from local files
- Full chunk data accessible without querying Milvus

Directory structure::

    data/chunks/{doc_id}.json

Each JSON file contains the full list of chunks for that document,
including content, page info, time info, and chunk metadata.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import get_project_root

logger = logging.getLogger(__name__)

CHUNKS_DIR = "data/chunks"


# ── Helpers ─────────────────────────────────────────────────────────

def _chunks_root() -> Path:
    return get_project_root() / CHUNKS_DIR


def _chunk_path(doc_id: str) -> Path:
    return _chunks_root() / f"{doc_id}.json"


# ── Save / Load / Delete ───────────────────────────────────────────

def save_chunks(
    doc_id: str,
    folder_id: str,
    chunks: list[Any],
    source_type: str = "upload",
    chunk_ids: list[str] | None = None,
) -> Path:
    """Save chunk results to a JSON file named ``{doc_id}.json``.

    *chunks* is a list of ``Chunk`` dataclass instances (from app.rag.chunker).

    Returns the path to the saved JSON file.
    """
    root = _chunks_root()
    root.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()

    chunk_list: list[dict[str, Any]] = []
    for i, c in enumerate(chunks):
        chunk_data: dict[str, Any] = {
            "chunk_index": c.index,
            "content": c.content,
            "start_time": c.start_time if c.start_time else 0.0,
            "end_time": c.end_time if c.end_time else 0.0,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "metadata": c.metadata if c.metadata else {},
        }
        if chunk_ids and i < len(chunk_ids):
            chunk_data["chunk_id"] = chunk_ids[i]
        chunk_list.append(chunk_data)

    doc_chunks: dict[str, Any] = {
        "document_id": doc_id,
        "folder_id": folder_id,
        "source_type": source_type,
        "chunk_count": len(chunk_list),
        "chunks": chunk_list,
        "created_at": now,
    }

    path = _chunk_path(doc_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc_chunks, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d chunks for document %s to %s", len(chunk_list), doc_id, path)
    return path


def load_chunks(doc_id: str) -> dict | None:
    """Load chunk data from its JSON file.

    Returns the full dict with ``document_id``, ``folder_id``,
    ``source_type``, ``chunk_count``, and ``chunks`` list.
    Returns None if the file doesn't exist.
    """
    path = _chunk_path(doc_id)
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning("Failed to read chunk file %s: %s", path, e)
        return None


def delete_chunks(doc_id: str) -> bool:
    """Delete the chunk JSON file for a document."""
    path = _chunk_path(doc_id)
    if path.exists():
        path.unlink()
        logger.info("Deleted chunk file for document %s", doc_id)
        return True
    return False


def list_all_chunk_files() -> list[str]:
    """List all document IDs that have saved chunk files."""
    root = _chunks_root()
    if not root.exists():
        return []
    return [p.stem for p in root.glob("*.json")]


def get_chunk_contents(doc_id: str) -> list[dict] | None:
    """Convenience: load just the chunks list for a document."""
    data = load_chunks(doc_id)
    if data is None:
        return None
    return data.get("chunks", [])
