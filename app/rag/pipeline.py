"""
RAG pipeline orchestrator — ties retrieval, reranking, and generation together.

Usage::

    pipeline = RAGPipeline(db_session)
    # blocking
    answer, sources = pipeline.query("What is justification?", kb_ids=["..."])
    # streaming
    for token in pipeline.query_stream("What is justification?", kb_ids=["..."]):
        print(token, end="")
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import get, get_rag_config
from app.db.models import KnowledgeBase, ModelRegistry
from app.models_manager.registry import get_embedding_function
from app.rag.generator import build_rag_messages, generate, generate_stream
from app.rag.reranker import rerank
from app.rag.retriever import search_vectors, search_bm25, reciprocal_rank_fusion
from app.services.llm_switch_service import get_active_vllm_model_name

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end retrieve → rerank → generate pipeline."""

    def __init__(self, db: Session):
        self.db = db

    # ── helpers ──────────────────────────────────────────────────────

    def _resolve_kbs(self, kb_ids: list[str]) -> list[KnowledgeBase]:
        return self.db.query(KnowledgeBase).filter(KnowledgeBase.id.in_(kb_ids)).all()

    def _get_default_llm(self) -> ModelRegistry | None:
        return (
            self.db.query(ModelRegistry)
            .filter(ModelRegistry.model_type == "llm", ModelRegistry.is_default == True)
            .first()
        )

    def _get_default_reranker(self) -> ModelRegistry | None:
        return (
            self.db.query(ModelRegistry)
            .filter(ModelRegistry.model_type == "reranker", ModelRegistry.is_default == True)
            .first()
        )

    # ── retrieve ────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        kb_ids: list[str],
        top_k: int | None = None,
        metadata_filters: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Encode query and search across specified knowledge bases using Hybrid Retrieval."""
        cfg = get_rag_config()
        k = top_k or cfg["top_k_retrieval"]
        kbs = self._resolve_kbs(kb_ids)

        if not kbs:
            logger.warning("No knowledge bases found for ids: %s", kb_ids)
            return []

        all_hits: list[dict[str, Any]] = []
        for kb in kbs:
            if not kb.collection_name:
                continue
            
            dense_hits = []
            sparse_hits = []

            # 1. Dense Search (Milvus)
            embed_fn = get_embedding_function(self.db, kb.embedding_model_id)
            if embed_fn is not None:
                try:
                    query_vector = embed_fn(query)
                    dense_hits = search_vectors(
                        kb.collection_name, query_vector, top_k=k
                    )
                except Exception as e:
                    logger.error("Vector search failed for KB '%s': %s", kb.name, e)
            else:
                logger.warning("No embedding model for KB '%s'", kb.name)

            # 2. Sparse Search (BM25 via SQLite FTS5)
            try:
                sparse_hits = search_bm25(self.db, query, kb_id=kb.id, top_k=k, metadata_filters=metadata_filters)
            except Exception as e:
                logger.error("BM25 search failed: %s", e)

            # 3. Reciprocal Rank Fusion
            hits = reciprocal_rank_fusion(dense_hits, sparse_hits, k=60)

            if hits:
                from app.db.models import ChunkMetadata, Document, Folder
                chunk_ids = [h["id"] for h in hits if "id" in h]
                
                query_obj = (
                    self.db.query(ChunkMetadata, Document)
                    .join(Document, ChunkMetadata.document_id == Document.id)
                    .join(Folder, Document.folder_id == Folder.id)
                    .filter(ChunkMetadata.id.in_(chunk_ids))
                    .filter(Folder.knowledge_base_id == kb.id)
                )
                
                # Apply explicit SQL filtering if there are metadata_filters
                if metadata_filters:
                    for k_filter, v_filter in metadata_filters.items():
                        if k_filter == "source_type":
                            # Map frontend source_type selections to document types / youtube source types
                            v_list = [v_filter] if isinstance(v_filter, str) else list(v_filter)
                            has_youtube = "youtube" in v_list
                            file_exts = [x for x in v_list if x != "youtube"]
                            
                            or_conds = []
                            if has_youtube:
                                or_conds.append(ChunkMetadata.source_type == "youtube")
                                or_conds.append(Document.source_type == "youtube")
                            if file_exts:
                                or_conds.append(Document.file_type.in_(file_exts))
                            
                            if or_conds:
                                from sqlalchemy import or_
                                query_obj = query_obj.filter(or_(*or_conds))
                        else:
                            if hasattr(ChunkMetadata, k_filter):
                                col_attr = getattr(ChunkMetadata, k_filter)
                                if isinstance(v_filter, list | tuple | set):
                                    query_obj = query_obj.filter(col_attr.in_(v_filter))
                                else:
                                    query_obj = query_obj.filter(col_attr == v_filter)
                            elif hasattr(Document, k_filter):
                                col_attr = getattr(Document, k_filter)
                                if isinstance(v_filter, list | tuple | set):
                                    query_obj = query_obj.filter(col_attr.in_(v_filter))
                                else:
                                    query_obj = query_obj.filter(col_attr == v_filter)

                chunks_data = query_obj.all()

                chunk_map = {}
                for c, d in chunks_data:
                    chunk_map[c.id] = {
                        "chunk_id": c.id,
                        "document_id": c.document_id,
                        "document_title": d.title or "Unknown",
                        "source_url": d.source_url,
                        "chunk_index": c.chunk_index,
                        "content": c.content,
                        "source_type": c.source_type,
                        "start_time": c.start_time,
                        "end_time": c.end_time,
                        "page_start": c.page_start,
                        "page_end": c.page_end,
                    }

                for h in hits:
                    chunk_id = h.get("id")
                    if chunk_id in chunk_map:
                        merged = {"score": h["score"], "kb_id": kb.id, "kb_name": kb.name}
                        merged.update(chunk_map[chunk_id])
                        all_hits.append(merged)
        all_hits.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_hits[:k]

    # ── full pipeline (blocking) ────────────────────────────────────

    def query(
        self,
        user_message: str,
        plan: dict,
        kb_ids: list[str],
        system_prompt: str | None = None,
        chat_history: list[dict[str, str]] | None = None,
        user_memories: list | None = None,
        model_id: str | None = None,
        images: list[str] | None = None,
        metadata_filters: dict | None = None,
    ) -> tuple[str, list[dict]]:
        """Run full RAG pipeline: multi-step retrieve → deduplicate → rerank → generate.

        Returns ``(answer_text, sources_list)``.
        """
        cfg = get_rag_config()

        sub_queries = plan.get("sub_queries", [user_message])
        
        # Merge plan filters with frontend filters
        plan_filters = plan.get("metadata_filters", {}) or {}
        explicit_filters = metadata_filters or {}
        merged_filters = {**plan_filters, **explicit_filters}
        
        user_intent = plan.get("user_intent", "General")

        # 1. Retrieve & Deduplicate
        all_retrieved = []
        seen_chunks = set()

        for sq in sub_queries:
            retrieved = self.retrieve(sq, kb_ids, metadata_filters=merged_filters)
            for chunk in retrieved:
                cid = chunk.get("chunk_id") or chunk.get("id")
                if cid and cid not in seen_chunks:
                    seen_chunks.add(cid)
                    all_retrieved.append(chunk)

        if not all_retrieved:
            return "I couldn't find any relevant information in the selected knowledge bases.", []

        # 2. Rerank first (scores non-expanded chunks)
        reranker_model = self._get_default_reranker()
        if reranker_model and reranker_model.model_path:
            reranked = rerank(
                user_message, all_retrieved,
                top_k=cfg["top_k_rerank"],
                model_path=reranker_model.model_path,
            )
        else:
            all_retrieved.sort(key=lambda x: x.get("score", 0), reverse=True)
            reranked = all_retrieved[: cfg["top_k_rerank"]]

        # 3. Autocut Threshold & Gap-based Cutoff
        has_rerank = any("rerank_score" in c for c in reranked)
        score_key = "rerank_score" if has_rerank else "score"
        if has_rerank:
            threshold = cfg.get("reranker_threshold", 0.5)
        else:
            threshold = 0.01  # Fallback for RRF ranking scores
            
        above_threshold = [
            c for c in reranked 
            if c.get(score_key, 0.0) >= threshold
        ]
        
        surviving_chunks = []
        if above_threshold:
            max_score = above_threshold[0].get(score_key, 0.0)
            prev_score = None
            for c in above_threshold:
                score = c.get(score_key, 0.0)
                if max_score > 0 and score < 0.35 * max_score:
                    break
                if prev_score is not None and prev_score > 0 and score < 0.5 * prev_score:
                    break
                surviving_chunks.append(c)
                prev_score = score

        if not surviving_chunks:
            return "I couldn't find any highly relevant information in the knowledge base.", []

        # 4. Expand surviving chunks
        expanded_chunks = self._expand_with_adjacent_chunks(surviving_chunks)

        # 5. Build messages
        base_prompt = system_prompt or cfg["system_prompt"]
        
        memory_text = ""
        if user_memories:
            m_lines = [f"- {m.memory_text}" for m in user_memories]
            memory_text = "Relevant Memories:\n" + "\n".join(m_lines) + "\n\n"
            
        intent_text = f"User Intent: {user_intent}\n\n"
        
        prompt = f"{base_prompt}\n\n{memory_text}{intent_text}"

        messages = build_rag_messages(user_message, expanded_chunks, prompt, chat_history, images=images)

        # 6. Generate
        llm_model = self._resolve_llm(model_id)
        active_model = get_active_vllm_model_name()
        answer = generate(
            messages,
            max_new_tokens=cfg["max_new_tokens"],
            temperature=cfg["temperature"],
            model_path=active_model or (llm_model.model_path if llm_model else None),
        )

        # 7. Build sources
        sources = self._format_sources(expanded_chunks)

        return answer, sources

    # ── streaming pipeline ──────────────────────────────────────────

    def query_stream(
        self,
        user_message: str,
        plan: dict,
        kb_ids: list[str],
        system_prompt: str | None = None,
        chat_history: list[dict[str, str]] | None = None,
        user_memories: list | None = None,
        model_id: str | None = None,
        images: list[str] | None = None,
        metadata_filters: dict | None = None,
    ) -> Generator[str, None, list[dict]]:
        """Streaming variant — yields tokens, returns sources on StopIteration."""
        cfg = get_rag_config()

        sub_queries = plan.get("sub_queries", [user_message])
        
        # Merge plan filters with frontend filters
        plan_filters = plan.get("metadata_filters", {}) or {}
        explicit_filters = metadata_filters or {}
        merged_filters = {**plan_filters, **explicit_filters}
        
        user_intent = plan.get("user_intent", "General")

        # 1. Retrieve & Deduplicate
        all_retrieved = []
        seen_chunks = set()

        for sq in sub_queries:
            retrieved = self.retrieve(sq, kb_ids, metadata_filters=merged_filters)
            for chunk in retrieved:
                cid = chunk.get("chunk_id") or chunk.get("id")
                if cid and cid not in seen_chunks:
                    seen_chunks.add(cid)
                    all_retrieved.append(chunk)

        if not all_retrieved:
            yield "I couldn't find any relevant information in the selected knowledge bases."
            return []

        # 2. Rerank first (scores non-expanded chunks)
        reranker_model = self._get_default_reranker()
        if reranker_model and reranker_model.model_path:
            reranked = rerank(
                user_message, all_retrieved,
                top_k=cfg["top_k_rerank"],
                model_path=reranker_model.model_path,
            )
        else:
            all_retrieved.sort(key=lambda x: x.get("score", 0), reverse=True)
            reranked = all_retrieved[: cfg["top_k_rerank"]]

        # 3. Autocut Threshold & Gap-based Cutoff
        has_rerank = any("rerank_score" in c for c in reranked)
        score_key = "rerank_score" if has_rerank else "score"
        if has_rerank:
            threshold = cfg.get("reranker_threshold", 0.5)
        else:
            threshold = 0.01  # Fallback for RRF ranking scores
            
        above_threshold = [
            c for c in reranked 
            if c.get(score_key, 0.0) >= threshold
        ]
        
        surviving_chunks = []
        if above_threshold:
            max_score = above_threshold[0].get(score_key, 0.0)
            prev_score = None
            for c in above_threshold:
                score = c.get(score_key, 0.0)
                if max_score > 0 and score < 0.35 * max_score:
                    break
                if prev_score is not None and prev_score > 0 and score < 0.5 * prev_score:
                    break
                surviving_chunks.append(c)
                prev_score = score

        if not surviving_chunks:
            yield "I couldn't find any highly relevant information in the knowledge base."
            return []

        # 4. Expand surviving chunks
        expanded_chunks = self._expand_with_adjacent_chunks(surviving_chunks)

        # 5. Build messages
        base_prompt = system_prompt or cfg["system_prompt"]
        
        memory_text = ""
        if user_memories:
            m_lines = [f"- {m.memory_text}" for m in user_memories]
            memory_text = "Relevant Memories:\n" + "\n".join(m_lines) + "\n\n"
            
        intent_text = f"User Intent: {user_intent}\n\n"
        
        prompt = f"{base_prompt}\n\n{memory_text}{intent_text}"

        messages = build_rag_messages(user_message, expanded_chunks, prompt, chat_history, images=images)

        # 6. Stream generation
        llm_model = self._resolve_llm(model_id)
        active_model = get_active_vllm_model_name()
        for token in generate_stream(
            messages,
            max_new_tokens=cfg["max_new_tokens"],
            temperature=cfg["temperature"],
            model_path=active_model or (llm_model.model_path if llm_model else None),
        ):
            yield token

        return self._format_sources(expanded_chunks)

    # ── private helpers ─────────────────────────────────────────────

    def _expand_with_adjacent_chunks(self, chunks: list[dict]) -> list[dict]:
        """Fetch adjacent chunks and merge them into a single expanded text block."""
        if not chunks:
            return chunks

        from sqlalchemy import and_, or_
        from app.db.models import ChunkMetadata

        adj_conditions = []
        for c in chunks:
            if "document_id" in c and "chunk_index" in c:
                adj_conditions.append(
                    and_(
                        ChunkMetadata.document_id == c["document_id"],
                        ChunkMetadata.chunk_index.in_([c["chunk_index"] - 1, c["chunk_index"] + 1])
                    )
                )

        adj_map = {}
        if adj_conditions:
            adj_chunks = self.db.query(ChunkMetadata).filter(or_(*adj_conditions)).all()
            for ac in adj_chunks:
                adj_map[(ac.document_id, ac.chunk_index)] = ac.content

        expanded_chunks = []
        for c in chunks:
            chunk = c.copy()
            if "document_id" in chunk and "chunk_index" in chunk:
                doc_id = chunk["document_id"]
                c_idx = chunk["chunk_index"]

                prev_content = adj_map.get((doc_id, c_idx - 1))
                next_content = adj_map.get((doc_id, c_idx + 1))

                # Merge into one plain text block — no internal labels
                parts = []
                if prev_content:
                    parts.append(prev_content)
                parts.append(chunk.get("content", ""))
                if next_content:
                    parts.append(next_content)

                chunk["content"] = "\n\n".join(parts)
            
            expanded_chunks.append(chunk)

        return expanded_chunks

    def _resolve_llm(self, model_id: str | None) -> ModelRegistry | None:
        if model_id:
            return self.db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
        return self._get_default_llm()

    @staticmethod
    def _format_sources(chunks: list[dict]) -> list[dict]:
        sources = []
        for c in chunks:
            source = {
                "document_title": c.get("document_title", ""),
                "source_url": c.get("source_url", ""),
                "content": c.get("content", ""),
                "score": c.get("rerank_score") or c.get("score", 0),
                "kb_name": c.get("kb_name", ""),
                "source_type": c.get("source_type", ""),
            }
            if c.get("start_time") is not None:
                source["start_time"] = c["start_time"]
                source["end_time"] = c.get("end_time")
            if c.get("page_start") and c["page_start"] > 0:
                source["page_start"] = c["page_start"]
                source["page_end"] = c.get("page_end", c["page_start"])
            sources.append(source)
        return sources
