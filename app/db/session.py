"""
Database session management.

Uses synchronous SQLAlchemy with SQLite.  A lightweight ``get_db()``
dependency yields a session per request.

# WAL mode enabled for better concurrency. Still single-writer;
# timeout=30s with retry logic for transient lock errors.
"""

import functools
import logging
import sqlite3
import time
from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_database_url, get_project_root
from app.db.models import Base

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None


def _set_sqlite_pragmas(dbapi_conn, connection_record):
    """Enable WAL mode and other performance pragmas on every new connection."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.execute("PRAGMA busy_timeout=30000;")  # 30 seconds
    cursor.close()


def _get_engine():
    global _engine
    if _engine is None:
        url = get_database_url()
        # Ensure data directory exists for SQLite
        if url.startswith("sqlite"):
            db_path = url.replace("sqlite:///", "").lstrip("./")
            abs_db_path = get_project_root() / db_path
            abs_db_path.parent.mkdir(parents=True, exist_ok=True)
            # Use absolute path to avoid cwd issues in notebooks/subdirs
            url = f"sqlite:///{abs_db_path.as_posix()}"
        connect_args = {"check_same_thread": False, "timeout": 30.0} if "sqlite" in url else {}
        _engine = create_engine(url, connect_args=connect_args, echo=False)

        # Register WAL mode pragma for all new connections
        if "sqlite" in url:
            event.listen(_engine, "connect", _set_sqlite_pragmas)

    return _engine

def get_session_factory() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_get_engine())
    return _SessionLocal


def ensure_fts_table_exists(engine) -> None:
    """Create and backfill the FTS virtual table if missing."""
    with engine.connect() as conn:
        conn.execute(text("CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(content, chunk_id UNINDEXED);"))
        
        # Check if empty
        result = conn.execute(text("SELECT COUNT(*) FROM chunk_fts")).scalar()
        if result == 0:
            conn.execute(text("INSERT INTO chunk_fts (chunk_id, content) SELECT id, content FROM chunks;"))
        conn.commit()


def insert_into_fts(db: Session, chunk_id: str, content: str) -> None:
    """Insert a single chunk into the FTS5 index.

    Should be called within the same transaction as the chunks table insert
    so both are atomic.
    """
    db.execute(
        text("INSERT INTO chunk_fts(chunk_id, content) VALUES(:cid, :content)"),
        {"cid": chunk_id, "content": content},
    )


def insert_batch_into_fts(db: Session, items: list[tuple[str, str]]) -> None:
    """Insert multiple chunks into the FTS5 index in a single batch.

    *items* is a list of ``(chunk_id, content)`` tuples.
    Should be called within the same transaction as the chunks table inserts.
    """
    if not items:
        return
    for chunk_id, content in items:
        db.execute(
            text("INSERT INTO chunk_fts(chunk_id, content) VALUES(:cid, :content)"),
            {"cid": chunk_id, "content": content},
        )


def delete_from_fts(db: Session, chunk_ids: list[str]) -> None:
    """Remove chunks from the FTS5 index by chunk_id."""
    if not chunk_ids:
        return
    for cid in chunk_ids:
        db.execute(
            text("DELETE FROM chunk_fts WHERE chunk_id = :cid"),
            {"cid": cid},
        )


def sync_fts_index(db: Session) -> dict[str, int]:
    """Sync the FTS5 index with the chunks table by rebuilding it from scratch.
    
    This is extremely fast and avoids slow nested loop joins on unindexed columns.
    """
    # 1. Get old count of FTS rows
    old_fts_count = db.execute(text("SELECT COUNT(*) FROM chunk_fts")).scalar() or 0
    
    # 2. Clear the FTS5 table
    db.execute(text("DELETE FROM chunk_fts"))
    
    # 3. Copy all chunks from the main table
    db.execute(text("INSERT INTO chunk_fts (chunk_id, content) SELECT id, content FROM chunks"))
    
    # 4. Get new count
    new_fts_count = db.execute(text("SELECT COUNT(*) FROM chunk_fts")).scalar() or 0
    chunks_count = db.execute(text("SELECT COUNT(*) FROM chunks")).scalar() or 0
    
    db.commit()
    
    # Calculate stats
    deleted_orphans = max(0, old_fts_count - (chunks_count - (chunks_count - new_fts_count)))  # Approximate statistics for output info
    inserted_missing = max(0, chunks_count - old_fts_count)
    
    return {
        "deleted_orphans": old_fts_count, 
        "inserted_missing": chunks_count,
        "total_synced": new_fts_count
    }


def init_db() -> None:
    """Create all tables that don't yet exist."""
    engine = _get_engine()
    Base.metadata.create_all(bind=engine)
    ensure_fts_table_exists(engine)


def get_db():
    """FastAPI dependency — yields a DB session, auto-closes after request."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def retry_on_locked(max_retries: int = 3, base_delay: float = 0.1):
    """Decorator that retries a function on SQLite 'database is locked' errors.

    Uses exponential backoff: 0.1s, 0.2s, 0.4s.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "database is locked" in str(e).lower():
                        last_exc = e
                        if attempt < max_retries:
                            delay = base_delay * (2 ** attempt)
                            logger.warning(
                                "SQLite locked (attempt %d/%d), retrying in %.2fs: %s",
                                attempt + 1, max_retries, delay, e,
                            )
                            time.sleep(delay)
                        continue
                    raise
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator
