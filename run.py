"""
Entry point for the RAG Knowledge Base System.

Usage:
    python run_app.py
"""

import uvicorn
from app.core.config import load_config
from model_orchestrator import start_services, stop_services

if __name__ == "__main__":

    cfg = load_config()
    app_cfg = cfg.get("app", {})
    try :
        start_services()
        uvicorn.run(
            "app.main:create_app",
            factory=True,
            host=app_cfg.get("host", "0.0.0.0"),
            port=app_cfg.get("port", 8000),
            reload=app_cfg.get("debug", False),
        )
    finally:
        stop_services()
        pass
