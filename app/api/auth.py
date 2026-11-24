"""
Auth API — login, register, user info.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.db.models import User
from app.services.auth_service import (
    create_user,
    get_user_by_email,
    get_user_by_username,
    login_for_token,
)
from app.core.limiter import limiter

router = APIRouter()


# ── Schemas ─────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    display_name: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    display_name: str | None
    role: str
    custom_role_id: str | None = None
    custom_role_name: str | None = None
    is_active: bool

    class Config:
        from_attributes = True


# ── Endpoints ───────────────────────────────────────────────────────

@router.post("/login", response_model=TokenResponse)
@limiter.limit("5/minute")
def login(request: Request, body: LoginRequest, db: Session = Depends(get_db)):
    result = login_for_token(db, body.username, body.password)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    return result


@router.post("/register", response_model=UserResponse, status_code=201)
@limiter.limit("5/minute")
def register(request: Request, body: RegisterRequest, db: Session = Depends(get_db)):
    if get_user_by_username(db, body.username):
        raise HTTPException(status_code=400, detail="Username already taken")
    if get_user_by_email(db, body.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    user = create_user(
        db,
        username=body.username,
        email=body.email,
        password=body.password,
        display_name=body.display_name,
    )
    return user


@router.get("/me")
def get_me(user: User = Depends(get_current_user)):
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "display_name": user.display_name,
        "role": user.role,
        "custom_role_id": user.custom_role_id,
        "custom_role_name": user.custom_role.name if user.custom_role else None,
        "is_active": user.is_active,
    }
