"""
Configuration loader for the RAG Knowledge Base System.

Loads settings from config.json at project root. Settings can be
overridden at runtime via the admin API and persisted back to the file.
"""

import copy
import json
import logging
import os
import secrets
from pathlib import Path
from typing import Any

# ── Locate project root (directory containing config.json) ──────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.json"

# ── In-memory cache ─────────────────────────────────────────────────
_config: dict | None = None

logger = logging.getLogger(__name__)

# Keys injected at runtime from env vars — never persisted to config.json.
_SENSITIVE_PATHS: list[tuple[str, ...]] = [
    ("auth", "secret_key"),
    ("superadmin", "password"),
]


def _inject_runtime_secrets(cfg: dict) -> None:
    """Override sensitive config values from environment variables.

    If the corresponding env var is not set, a cryptographically-secure
    random value is generated for the current process lifetime.  These
    runtime-only values are stripped before ``save_config()`` writes to disk.
    """
    # ── Auth secret key ─────────────────────────────────────────
    env_secret = os.environ.get("RAG_SECRET_KEY")
    cfg.setdefault("auth", {})
    if env_secret:
        cfg["auth"]["secret_key"] = env_secret
    else:
        cfg["auth"]["secret_key"] = secrets.token_hex(32)
        logger.warning(
            "RAG_SECRET_KEY not set. Using ephemeral random key. "
            "Sessions will invalidate on restart. Set RAG_SECRET_KEY for persistence."
        )

    # ── Superadmin password ─────────────────────────────────────
    env_pwd = os.environ.get("RAG_SUPERADMIN_PASSWORD")
    cfg.setdefault("superadmin", {})
    if env_pwd:
        cfg["superadmin"]["password"] = env_pwd
    else:
        temp_pwd = secrets.token_urlsafe(16)
        cfg["superadmin"]["password"] = temp_pwd
        print(
            f"WARNING: Superadmin password not set in environment. "
            f"Generated temporary password: {temp_pwd}. "
            f"Change immediately via admin panel or restart with "
            f"RAG_SUPERADMIN_PASSWORD env var."
        )


def _strip_sensitive(cfg: dict) -> dict:
    """Return a deep copy of *cfg* with runtime-only secrets removed."""
    safe = copy.deepcopy(cfg)
    for key_path in _SENSITIVE_PATHS:
        d = safe
        for k in key_path[:-1]:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                break
        else:
            d.pop(key_path[-1], None)
    return safe


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (mutates base)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Path | str | None = None, *, reload: bool = False) -> dict:
    """Load configuration from JSON file, caching the result.

    Args:
        path: Optional path to config file (defaults to project root config.json).
        reload: Force re-read from disk.
    """
    global _config
    if _config is not None and not reload:
        return _config

    config_path = Path(path) if path else _CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        _config = json.load(f)

    _inject_runtime_secrets(_config)
    return _config


def get_config() -> dict:
    """Return the currently loaded configuration (loads if needed)."""
    if _config is None:
        return load_config()
    return _config


def get(key: str, default: Any = None) -> Any:
    """Get a dot-separated config value.  e.g. ``get("rag.chunk_size")``."""
    cfg = get_config()
    keys = key.split(".")
    current = cfg
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return default
    return current


def update_config(updates: dict, *, persist: bool = True) -> dict:
    """Deep-merge *updates* into the running config and optionally save to disk."""
    global _config
    cfg = get_config()
    _deep_merge(cfg, updates)

    if persist:
        save_config()

    return cfg


def save_config(path: Path | str | None = None) -> None:
    """Write the current in-memory config back to disk.

    Runtime-only secrets (JWT key, superadmin password) are stripped
    before writing so they are never persisted in plaintext.
    """
    config_path = Path(path) if path else _CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    safe_cfg = _strip_sensitive(get_config())
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(safe_cfg, f, indent=2, ensure_ascii=False)


def get_project_root() -> Path:
    return _PROJECT_ROOT


def get_database_url() -> str:
    return get("database.url", "sqlite:///./data/rag.db")


def get_milvus_config() -> dict:
    return {
        "host": get("milvus.host", "localhost"),
        "port": get("milvus.port", 19530),
        "metric_type": get("milvus.default_metric_type", "L2"),
        "index_type": get("milvus.index_type", "GPU_IVF_PQ"),
    }


def get_auth_config() -> dict:
    return {
        "secret_key": get("auth.secret_key", "change-me"),
        "algorithm": get("auth.algorithm", "HS256"),
        "expire_minutes": get("auth.access_token_expire_minutes", 1440),
    }


def get_rag_config() -> dict:
    return {
        "chunk_size": get("rag.chunk_size", 512),
        "chunk_overlap": get("rag.chunk_overlap", 50),
        "top_k_retrieval": get("rag.top_k_retrieval", 20),
        "top_k_rerank": get("rag.top_k_rerank", 5),
        "system_prompt": get("rag.system_prompt", "You are a helpful assistant."),
        "temperature": get("rag.temperature", 0.7),
        "max_new_tokens": get("rag.max_new_tokens", 2048),
        "reranker_threshold": get("rag.reranker_threshold", 0.5),
    }


def get_storage_config() -> dict:
    root = get_project_root()
    return {
        "upload_dir": str(root / get("storage.upload_dir", "data/uploads")),
        "models_dir": str(root / get("storage.models_dir", "data/models")),
        "max_file_size_mb": get("storage.max_file_size_mb", 100),
    }


