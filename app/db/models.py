"""
SQLAlchemy ORM models for the RAG Knowledge Base System.

Tables
------
- users              – authentication & roles
- knowledge_bases    – top-level KB containers
- folders            – hierarchical folders inside a KB
- documents          – document metadata (file info, processing status)
- chat_sessions      – per-user conversation sessions
- chat_messages      – individual messages in a session
- model_registry     – LLM / embedding / reranker model catalog
- system_configs     – key-value runtime configuration store

Document chunks + vectors live in the Milvus vector database.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


# ── Helpers ─────────────────────────────────────────────────────────
def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Base ────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ROLES  (custom access-control roles created by superadmin)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Role(Base):
    __tablename__ = "roles"

    id = Column(String(36), primary_key=True, default=_uuid)
    name = Column(String(64), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    # relationships
    kb_access = relationship("RoleKBAccess", back_populates="role", cascade="all, delete-orphan")
    users = relationship("User", back_populates="custom_role")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ROLE ↔ KB ACCESS  (which KBs each role can access)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class RoleKBAccess(Base):
    __tablename__ = "role_kb_access"

    id = Column(String(36), primary_key=True, default=_uuid)
    role_id = Column(String(36), ForeignKey("roles.id", ondelete="CASCADE"), nullable=False)
    kb_id = Column(String(36), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=_utcnow)

    # relationships
    role = relationship("Role", back_populates="kb_access")
    knowledge_base = relationship("KnowledgeBase")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  USERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=_uuid)
    username = Column(String(64), unique=True, nullable=False)
    email = Column(String(128), unique=True, nullable=False)
    display_name = Column(String(128), nullable=True)
    hashed_password = Column(String(128), nullable=False)
    role = Column(String(16), default="user")  # "user" | "admin" | "superadmin"
    custom_role_id = Column(String(36), ForeignKey("roles.id", ondelete="SET NULL"), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    # relationships
    chat_sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    knowledge_bases = relationship("KnowledgeBase", back_populates="creator")
    custom_role = relationship("Role", back_populates="users")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  KNOWLEDGE BASES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id = Column(String(36), primary_key=True, default=_uuid)
    name = Column(String(128), nullable=False)
    description = Column(Text, nullable=True)
    is_public = Column(Boolean, default=True)
    embedding_model_id = Column(String(36), ForeignKey("model_registry.id"), nullable=True)
    collection_name = Column(String(128), nullable=True)  # Milvus collection
    vector_dimension = Column(Integer, default=2048)
    created_by = Column(String(36), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    # relationships
    folders = relationship("Folder", back_populates="knowledge_base", cascade="all, delete-orphan")
    creator = relationship("User", back_populates="knowledge_bases")
    embedding_model = relationship("ModelRegistry", foreign_keys=[embedding_model_id])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FOLDERS  (hierarchical tree inside a KB)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Folder(Base):
    __tablename__ = "folders"

    id = Column(String(36), primary_key=True, default=_uuid)
    knowledge_base_id = Column(String(36), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(128), nullable=False)
    description = Column(Text, nullable=True)
    parent_id = Column(String(36), ForeignKey("folders.id"), nullable=True)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    # relationships
    children = relationship(
        "Folder",
        backref="parent",
        remote_side=[id],
        cascade="all, delete-orphan",
        single_parent=True,
    )
    knowledge_base = relationship("KnowledgeBase", back_populates="folders")
    documents = relationship("Document", back_populates="folder", cascade="all, delete-orphan")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DOCUMENTS  (file metadata + processing status)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_status", "status"),
        Index("ix_documents_folder_id", "folder_id"),
    )

    id = Column(String(36), primary_key=True, default=_uuid)
    folder_id = Column(String(36), ForeignKey("folders.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(512), nullable=False)  # Unified document title (replaces original_filename + stored_filename)
    file_type = Column(String(32), nullable=False)
    source_type = Column(String(32), default="upload")  # "upload" | "youtube" | ...
    file_size = Column(BigInteger, default=0)
    status = Column(String(32), default="pending")  # "pending" | "processing" | "chunked" | "processed" | "error"
    error_message = Column(Text, nullable=True)
    chunk_count = Column(Integer, default=0)
    source_url = Column(String(1024), nullable=True)
    metadata_json = Column(Text, nullable=True)  # JSON-encoded extra metadata
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    # relationships
    folder = relationship("Folder", back_populates="documents")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CHAT SESSIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True, default=_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(512), default="New Chat")
    knowledge_base_ids = Column(Text, nullable=True)  # JSON-encoded list of KB ids
    model_id = Column(String(36), ForeignKey("model_registry.id"), nullable=True)  # which LLM
    system_prompt = Column(Text, nullable=True)  # per-session prompt override
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    # relationships
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
    )
    model = relationship("ModelRegistry", foreign_keys=[model_id])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CHAT MESSAGES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String(36), primary_key=True, default=_uuid)
    session_id = Column(String(36), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(16), nullable=False)  # "user" | "assistant" | "system"
    content = Column(Text, nullable=False)
    sources_json = Column(Text, nullable=True)  # JSON-encoded list of source references
    images_json = Column(Text, nullable=True)  # JSON-encoded list of base64 image data URIs
    token_count = Column(Integer, nullable=True)
    parent_id = Column(String(36), ForeignKey("chat_messages.id", ondelete="SET NULL"), nullable=True)  # original message this is an edit of
    version = Column(Integer, default=1)  # 1 = original, 2+ = edits
    is_active = Column(Boolean, default=True)  # only the active version is shown in conversation
    created_at = Column(DateTime, default=_utcnow)

    # relationships
    session = relationship("ChatSession", back_populates="messages")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MODEL REGISTRY  (LLM, embedding, reranker models)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(String(36), primary_key=True, default=_uuid)
    name = Column(String(128), nullable=False)
    model_type = Column(String(32), nullable=False)  # "llm" | "embedding" | "reranker"
    model_path = Column(String(512), nullable=True)  # local path or HuggingFace repo id
    api_endpoint = Column(String(512), nullable=True)  # if served externally
    vector_dimension = Column(Integer, nullable=True)  # for embedding models
    max_tokens = Column(Integer, nullable=True)
    status = Column(String(32), default="available")  # "available" | "loading" | "error" | "downloading"
    is_default = Column(Boolean, default=False)
    config_json = Column(Text, nullable=True)  # model-specific JSON config
    uploaded_by = Column(String(36), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    # relationships
    uploader = relationship("User", foreign_keys=[uploaded_by])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SYSTEM CONFIGS  (runtime key-value store, editable from admin UI)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SystemConfig(Base):
    __tablename__ = "system_configs"

    id = Column(String(36), primary_key=True, default=_uuid)
    key = Column(String(128), unique=True, nullable=False)
    value = Column(Text, nullable=False)  # JSON-encoded value
    description = Column(Text, nullable=True)
    category = Column(String(32), default="system")  # "rag" | "embedding" | "llm" | "ui" | "system"
    updated_by = Column(String(36), ForeignKey("users.id"), nullable=True)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CHUNKS 
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ChunkMetadata(Base):
    __tablename__ = "chunks"

    id = Column(String(36), primary_key=True, default=_uuid)
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    source_type = Column(String(32), nullable=True)
    start_time = Column(Integer, nullable=True)
    end_time = Column(Integer, nullable=True)
    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  USER MEMORY 
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class UserMemory(Base):
    __tablename__ = "user_memories"

    id = Column(String(36), primary_key=True, default=_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    memory_text = Column(Text, nullable=False)
    importance = Column(Integer, default=5)
    created_at = Column(DateTime, default=_utcnow)
    last_accessed_at = Column(DateTime, default=_utcnow)
    memory_type = Column(String(32), default="core")

    # relationships
    user = relationship("User")
