"""
Vector retriever — searches Milvus for relevant chunks.

Each KnowledgeBase has its own Milvus collection (``collection_name``).
The retriever encodes the query with the KB's embedding model and
returns top-k hits.
"""

from __future__ import annotations

import logging
from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from app.core.config import get_milvus_config, get_rag_config

logger = logging.getLogger(__name__)

# ── Milvus connection singleton ─────────────────────────────────────
_connected = False


def ensure_connection() -> None:
    global _connected
    if _connected:
        return
    cfg = get_milvus_config()
    connections.connect(alias="default", host=cfg["host"], port=cfg["port"])
    _connected = True
    logger.info("Connected to Milvus at %s:%s", cfg["host"], cfg["port"])


def disconnect() -> None:
    global _connected
    try:
        connections.disconnect(alias="default")
    except Exception:
        pass
    _connected = False


# ── Collection management ───────────────────────────────────────────

# TODO: Implement Milvus Partitioning
# For future scaling, we should use partitions (e.g., by month/week) so we can drop whole
# partitions to reclaim space cleanly rather than deleting individual document rows.
def create_collection(name: str, vector_dim: int, metric_type: str | None = None) -> Collection:
    """Create a Milvus collection for a knowledge base.

    Uses GPU_IVF_PQ index by default (configurable via milvus.index_type).
    """
    ensure_connection()
    cfg = get_milvus_config()
    metric = metric_type or cfg["metric_type"]
    index_type = cfg["index_type"]

    if utility.has_collection(name):
        return Collection(name)

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
    ]
    schema = CollectionSchema(fields=fields, description=f"KB collection: {name}")
    col = Collection(name=name, schema=schema)

    # Build GPU-accelerated index (matches milvus_db_test.ipynb)
    index_params = {
        "index_type": index_type,
        "metric_type": metric,
        "params": {"nlist": 1024},
    }
    
    if index_type == "GPU_IVF_PQ":
        index_params["params"]["m"] = 32        # 64-dim sub-vectors 
        index_params["params"]["nbits"] = 8
        
    col.create_index(field_name="vector", index_params=index_params)
    col.load()
    logger.info(
        "Created Milvus collection '%s' (dim=%d, metric=%s, index=%s)",
        name, vector_dim, metric, index_type,
    )
    return col


def drop_collection(name: str) -> None:
    ensure_connection()
    if utility.has_collection(name):
        utility.drop_collection(name)
        logger.info("Dropped Milvus collection '%s'", name)


def collection_exists(name: str) -> bool:
    ensure_connection()
    return utility.has_collection(name)


def get_collection(name: str) -> Collection:
    ensure_connection()
    col = Collection(name)
    col.load()
    return col


# ── Insert & Search ────────────────────────────────────────────────

def insert_vectors(
    collection_name: str,
    chunk_ids: list[str],
    vectors: list[list[float]],
) -> list[str]:
    """Insert chunk vectors into a Milvus collection. Returns inserted IDs."""
    ensure_connection()
    col = get_collection(collection_name)
    data = [chunk_ids, vectors]
    result = col.insert(data)
    logger.info("Inserted %d vectors into '%s'", len(vectors), collection_name)
    return result.primary_keys


def search_vectors(
    collection_name: str,
    query_vector: list[float],
    top_k: int | None = None,
    metric_type: str | None = None,
    metadata_filters: dict | None = None,
) -> list[dict[str, Any]]:
    """Search for the nearest vectors. Returns list of {id, score} dicts."""
    ensure_connection()
    cfg = get_rag_config()
    k = top_k or cfg["top_k_retrieval"]

    expr = None
    if metadata_filters:
        expr_parts = []
        for k_filter, v_filter in metadata_filters.items():
            if isinstance(v_filter, list | tuple | set):
                val_list = [f'"{v}"' if isinstance(v, str) else str(v) for v in v_filter]
                expr_parts.append(f'{k_filter} in [{", ".join(val_list)}]')
            elif isinstance(v_filter, str):
                expr_parts.append(f'{k_filter} == "{v_filter}"')
            else:
                expr_parts.append(f'{k_filter} == {v_filter}')
        if expr_parts:
            expr = " and ".join(expr_parts)

    col = get_collection(collection_name)
    results = col.search(
        data=[query_vector],
        anns_field="vector",
        param={
            "metric_type": metric_type or get_milvus_config()["metric_type"],
            "params": {"nprobe": 10},
        },
        limit=k,
        expr=expr,
        output_fields=["id"],
    )

    hits: list[dict[str, Any]] = []
    for hit in results[0]:
        hits.append({"id": str(hit.id), "score": hit.score})

    return hits


def get_collection_stats(collection_name: str) -> dict:
    """Return row count and other stats for a collection."""
    ensure_connection()
    if not utility.has_collection(collection_name):
        return {"exists": False}
    col = get_collection(collection_name)
    col.flush()
    return {
        "exists": True,
        "name": collection_name,
        "num_entities": col.num_entities,
    }


def search_bm25(
    db: Any,
    query: str,
    kb_id: str | None = None,
    top_k: int = 20,
    metadata_filters: dict | None = None,
) -> list[dict[str, Any]]:
    """Search for keywords using SQLite FTS5 table `chunk_fts`.
    
    If `kb_id` is supplied, filters search results to only return chunks belonging to that KB.
    """
    from sqlalchemy import text
    
    # 1. Clean punctuation, lowercase, and tokenize
    query_clean = "".join([c.lower() if (c.isalnum() or c.isspace()) else " " for c in query])
    tokens = [w for w in query_clean.split() if w]
    if not tokens:
        return []
        
    # 2. Stop words to filter (Indonesian + English)
    STOP_WORDS = {
        "dan", "yang", "di", "ke", "dari", "untuk", "dengan", "adalah", "itu", "ini", 
        "pada", "dalam", "oleh", "atau", "juga", "sebagai", "secara", "jika", "saat", 
        "tentang", "bagaimana", "mengapa", "apa", "siapa", "dimana", "ada", "saya", "kami",
        "and", "or", "but", "is", "are", "was", "were", "to", "of", "in", "for", "on", 
        "with", "at", "by", "an", "a", "the", "this", "that", "it", "from", "how", "why", 
        "what", "who", "where", "about", "we", "i"
    }
    
    # Filter tokens to keep only meaningful keywords
    keywords = [w for w in tokens if w not in STOP_WORDS]
    if not keywords:
        keywords = tokens
        
    # 3. Join with OR to search for any match, letting BM25 rank them
    clean_query = " OR ".join(keywords)
    
    from app.db.models import ChunkMetadata, Document
    
    clauses = ["fts.chunk_fts MATCH :query"]
    params = {"query": clean_query, "limit": top_k}
    
    if kb_id:
        clauses.append("f.knowledge_base_id = :kb_id")
        params["kb_id"] = kb_id
        
    def _build_in_clause(col_name: str, vals: list | tuple | set, query_params: dict, prefix: str) -> str:
        placeholders = []
        for idx, val in enumerate(vals):
            p_key = f"{prefix}_{idx}"
            placeholders.append(f":{p_key}")
            query_params[p_key] = val
        return f"{col_name} IN ({', '.join(placeholders)})"

    if metadata_filters:
        for k_filter, v_filter in metadata_filters.items():
            if k_filter == "source_type":
                v_list = [v_filter] if isinstance(v_filter, str) else list(v_filter)
                has_youtube = "youtube" in v_list
                file_exts = [x for x in v_list if x != "youtube"]
                
                sub_clauses = []
                if has_youtube:
                    sub_clauses.append("c.source_type = 'youtube'")
                    sub_clauses.append("d.source_type = 'youtube'")
                if file_exts:
                    in_clause = _build_in_clause("d.file_type", file_exts, params, "meta_file_types")
                    sub_clauses.append(in_clause)
                    
                if sub_clauses:
                    clauses.append(f"({' OR '.join(sub_clauses)})")
            else:
                if hasattr(ChunkMetadata, k_filter):
                    if isinstance(v_filter, list | tuple | set):
                        in_clause = _build_in_clause(f"c.{k_filter}", v_filter, params, f"meta_{k_filter}")
                        clauses.append(in_clause)
                    else:
                        p_name = f"meta_{k_filter}"
                        clauses.append(f"c.{k_filter} = :{p_name}")
                        params[p_name] = v_filter
                elif hasattr(Document, k_filter):
                    if isinstance(v_filter, list | tuple | set):
                        in_clause = _build_in_clause(f"d.{k_filter}", v_filter, params, f"meta_{k_filter}")
                        clauses.append(in_clause)
                    else:
                        p_name = f"meta_{k_filter}"
                        clauses.append(f"d.{k_filter} = :{p_name}")
                        params[p_name] = v_filter

    if kb_id or metadata_filters:
        sql = f"""
            SELECT fts.chunk_id, fts.rank 
            FROM chunk_fts fts
            JOIN chunks c ON c.id = fts.chunk_id
            JOIN documents d ON d.id = c.document_id
            JOIN folders f ON f.id = d.folder_id
            WHERE {" AND ".join(clauses)}
            ORDER BY fts.rank 
            LIMIT :limit
        """
    else:
        sql = """
            SELECT chunk_id, rank 
            FROM chunk_fts 
            WHERE chunk_fts MATCH :query 
            ORDER BY rank 
            LIMIT :limit
        """
        
    results = db.execute(text(sql), params).fetchall()
    
    hits = []
    for row in results:
        # FTS5 rank is more negative for better matches. 
        # Negate it so higher = better, matching typical vector logic.
        hits.append({"id": row.chunk_id, "score": -row.rank})
        
    return hits



def reciprocal_rank_fusion(dense_hits: list[dict], sparse_hits: list[dict], k: int = 60) -> list[dict]:
    """Fuse dense and sparse hits using Reciprocal Rank Fusion."""
    scores = {}
    
    dense_hits = sorted(dense_hits, key=lambda x: x.get("score", 0), reverse=True)
    sparse_hits = sorted(sparse_hits, key=lambda x: x.get("score", 0), reverse=True)
    
    for rank, hit in enumerate(dense_hits):
        chunk_id = hit["id"]
        if chunk_id not in scores:
            scores[chunk_id] = 0.0
        scores[chunk_id] += 1.0 / (k + rank + 1)
        
    for rank, hit in enumerate(sparse_hits):
        chunk_id = hit["id"]
        if chunk_id not in scores:
            scores[chunk_id] = 0.0
        scores[chunk_id] += 1.0 / (k + rank + 1)
        
    fused = [{"id": chunk_id, "score": score} for chunk_id, score in scores.items()]
    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused


def delete_vectors_by_ids(
    collection_name: str,
    chunk_ids: list[str],
) -> int:
    """Delete vectors from a Milvus collection by their primary key IDs.

    Used during document deletion to remove orphaned vectors.
    Returns the number of IDs targeted for deletion.
    """
    if not chunk_ids:
        return 0
    ensure_connection()
    if not utility.has_collection(collection_name):
        logger.warning("Collection '%s' does not exist — nothing to delete", collection_name)
        return 0

    col = get_collection(collection_name)
    id_list = ", ".join(f'"{cid}"' for cid in chunk_ids)
    expr = f"id in [{id_list}]"
    col.delete(expr)
    col.flush()
    logger.info("Deleted %d vectors from '%s'", len(chunk_ids), collection_name)
    return len(chunk_ids)


def delete_by_document_id(collection_name: str, doc_id: str, db: Session) -> int:
    """Query chunk IDs associated with doc_id from SQLite and delete them from Milvus."""
    from app.db.models import ChunkMetadata
    from sqlalchemy.orm import Session
    chunk_ids = [
        row.id for row in
        db.query(ChunkMetadata.id).filter(ChunkMetadata.document_id == doc_id).all()
    ]
    if not chunk_ids:
        return 0
    return delete_vectors_by_ids(collection_name, chunk_ids)

