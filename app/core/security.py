"""
Security utilities — JWT tokens and password hashing.

Uses bcrypt directly (passlib is incompatible with bcrypt ≥ 4.1).
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from jose import JWTError, jwt

from app.core.config import get_auth_config

# ── Password hashing ────────────────────────────────────────────────


def hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# ── JWT helpers ─────────────────────────────────────────────────────

def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    auth = get_auth_config()
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=auth["expire_minutes"])
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, auth["secret_key"], algorithm=auth["algorithm"])


def decode_access_token(token: str) -> dict[str, Any] | None:
    """Return the token payload or ``None`` if invalid / expired."""
    auth = get_auth_config()
    try:
        payload = jwt.decode(token, auth["secret_key"], algorithms=[auth["algorithm"]])
        return payload
    except JWTError:
        return None
