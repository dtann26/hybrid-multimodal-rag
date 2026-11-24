"""
Authentication service — user CRUD, login, registration.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.security import create_access_token, hash_password, verify_password
from app.db.models import User


def get_user_by_username(db: Session, username: str) -> User | None:
    return db.query(User).filter(User.username == username).first()


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: str) -> User | None:
    return db.query(User).filter(User.id == user_id).first()


def authenticate_user(db: Session, username: str, password: str) -> User | None:
    user = get_user_by_username(db, username)
    if user is None or not verify_password(password, user.hashed_password):
        return None
    if not user.is_active:
        return None
    return user


def create_user(
    db: Session,
    username: str,
    email: str,
    password: str,
    role: str = "user",
    display_name: str | None = None,
) -> User:
    user = User(
        username=username,
        email=email,
        hashed_password=hash_password(password),
        role=role,
        display_name=display_name or username,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def login_for_token(db: Session, username: str, password: str) -> dict | None:
    """Authenticate and return ``{"access_token": ..., "token_type": "bearer"}``."""
    user = authenticate_user(db, username, password)
    if user is None:
        return None
    token = create_access_token(data={"sub": user.id, "role": user.role})
    return {"access_token": token, "token_type": "bearer"}


def list_users(db: Session, skip: int = 0, limit: int = 50) -> dict:
    limit = min(max(1, limit), 100)
    skip = max(0, skip)
    total_count = db.query(User).count()
    users = db.query(User).offset(skip).limit(limit).all()
    
    # Convert users list to dictionaries
    items = [
        {
            "id": u.id,
            "username": u.username,
            "email": u.email,
            "display_name": u.display_name,
            "role": u.role,
            "custom_role_id": u.custom_role_id,
            "custom_role_name": u.custom_role.name if u.custom_role else None,
            "is_active": u.is_active,
            "created_at": u.created_at.isoformat() if u.created_at else None,
        }
        for u in users
    ]
    
    return {
        "items": items,
        "total": total_count,
        "skip": skip,
        "limit": limit,
        "has_more": skip + len(items) < total_count
    }


def update_user_role(db: Session, user_id: str, new_role: str, custom_role_id: str | None = None) -> User | None:
    user = get_user_by_id(db, user_id)
    if user is None:
        return None
    user.role = new_role
    user.custom_role_id = custom_role_id
    db.commit()
    db.refresh(user)
    return user


def deactivate_user(db: Session, user_id: str) -> bool:
    user = get_user_by_id(db, user_id)
    if user is None:
        return False
    user.is_active = False
    db.commit()
    return True


def delete_user(db: Session, user_id: str) -> bool:
    from app.db.models import User, KnowledgeBase
    from app.services.kb_service import delete_knowledge_base
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        return False
    
    # 1. Clean up KBs owned by this user (which cascades to deleting Milvus vectors)
    kbs = db.query(KnowledgeBase).filter(KnowledgeBase.created_by == user_id).all()
    for kb in kbs:
        delete_knowledge_base(db, kb.id)
    
    # 2. Delete the user (cascades to chat sessions & messages)
    db.delete(user)
    db.commit()
    return True
