"""
FastAPI application factory.

Creates the app, mounts routers, configures static files, templates,
CORS, and startup/shutdown lifecycle hooks.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from slowapi.errors import RateLimitExceeded

from app.core.config import get, load_config
from app.db.session import get_db, get_session_factory, init_db

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    # ── Startup ─────────────────────────────────────────────────
    load_config()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Starting RAG Knowledge Base System...")

    # Init database
    init_db()
    logger.info("Database initialised.")

    # Run seed
    from app.db.seed import run_seed
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        run_seed(db)
    finally:
        db.close()
    logger.info("Seed complete.")

    # Ensure upload directories exist
    upload_dir = Path(get("storage.upload_dir", "data/uploads"))
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Run LLM state diagnostic and failsafe in the background
    import asyncio
    from app.services.llm_switch_service import sync_llm_state_on_startup
    
    # We use asyncio.to_thread so the synchronous httpx call and potential
    # docker commands don't block the FastAPI event loop during startup.
    asyncio.create_task(asyncio.to_thread(sync_llm_state_on_startup))

    yield

    # ── Shutdown ────────────────────────────────────────────────
    logger.info("Shutting down...")
    from app.rag.retriever import disconnect
    disconnect()
    logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    app = FastAPI(
        title=get("app.title", "RAG Knowledge Base System"),
        version="1.0.0",
        lifespan=lifespan,
    )

    # ── Rate limiter ────────────────────────────────────────────
    # In-memory storage (resets on restart; no Redis needed for single-instance).
    # Per-route limits are applied via @limiter.limit() decorators in routers.
    from app.core.limiter import limiter
    app.state.limiter = limiter

    def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Try again later."},
        )

    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

    # ── CORS ────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── API routers ─────────────────────────────────────────────
    from app.api.auth import router as auth_router
    from app.api.chat import router as chat_router
    from app.api.knowledge_base import router as kb_router
    from app.api.documents import router as docs_router
    from app.api.admin import router as admin_router

    app.include_router(auth_router, prefix="/api/auth", tags=["Auth"])
    app.include_router(chat_router, prefix="/api/chat", tags=["Chat"])
    app.include_router(kb_router, prefix="/api/kb", tags=["Knowledge Base"])
    app.include_router(docs_router, prefix="/api/documents", tags=["Documents"])
    app.include_router(admin_router, prefix="/api/admin", tags=["Admin"])

    # ── Static files & templates ────────────────────────────────
    static_dir = APP_DIR / "static"
    static_dir.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

    # ── Page routes ─────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse("login.html", {"request": request})

    @app.get("/chat", response_class=HTMLResponse)
    async def chat_page(request: Request):
        return templates.TemplateResponse("chat.html", {"request": request})

    @app.get("/admin", response_class=HTMLResponse)
    async def admin_page(request: Request):
        return templates.TemplateResponse("admin.html", {"request": request})

    @app.get("/forgot-password", response_class=HTMLResponse)
    async def forgot_password_page(request: Request):
        return templates.TemplateResponse("forgot_password.html", {"request": request})

    # ── Health check ────────────────────────────────────────────
    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    return app
