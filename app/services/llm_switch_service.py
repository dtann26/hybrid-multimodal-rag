"""
LLM hot-swap service — manages switching the vLLM LLM container to a
different model at runtime without restarting the web application.

Supports exactly three hardcoded models:
  1. Qwen2.5-VL-7B-Instruct-NVFP4  (default)
  2. Phi-4-multimodal-instruct-NVFP4
  3. Gemma-4-31B-IT-NVFP4

Gemma requires a different vLLM Docker image and command arguments, so
the ``docker-compose.yml`` is rewritten dynamically before restarting.

Flow:
  1. Validate the target model is one of the three allowed models.
  2. Download the model via ``huggingface-cli`` if not present on disk.
  3. Back up the current ``.env`` and ``docker-compose.yml``.
  4. Update ``LLM_MODEL_DIR_NAME`` in the project's ``.env`` file.
  5. Rewrite the ``vllm-llm`` service in ``docker-compose.yml`` for the
     target model (Gemma uses a different image/config).
  6. ``docker compose stop vllm-llm``  (free VRAM).
  7. ``docker compose up -d vllm-llm`` (start with new model).
  8. Poll ``http://localhost:8001/v1/models`` until 200 OK (10-min timeout).
  9. Update ``models.default_llm`` in config.json so the app sends
     requests to the right ``--served-model-name``.
  10. On any failure/timeout → automatic rollback of ``.env``,
      ``docker-compose.yml``, and the container.
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from enum import Enum
from pathlib import Path
from typing import Any

import httpx

from app.core.config import get_project_root, update_config

logger = logging.getLogger(__name__)

# ── Allowed models (hardcoded) ──────────────────────────────────────

ALLOWED_MODELS: set[str] = {
    "Qwen2.5-VL-7B-Instruct-NVFP4",
    "Phi-4-multimodal-instruct-NVFP4",
    "Gemma-4-31B-IT-NVFP4",
}

# ── Download commands per model ─────────────────────────────────────

_DOWNLOAD_COMMANDS: dict[str, list[str]] = {
    "Qwen2.5-VL-7B-Instruct-NVFP4": [
        "huggingface-cli", "download",
        "nvidia/Qwen2.5-VL-7B-Instruct-NVFP4",
        "--local-dir", "./models/Qwen2.5-VL-7B-Instruct-NVFP4",
        "--local-dir-use-symlinks", "False",
    ],
    "Phi-4-multimodal-instruct-NVFP4": [
        "huggingface-cli", "download",
        "nvidia/Phi-4-multimodal-instruct-NVFP4",
        "--local-dir", "./models/Phi-4-multimodal-instruct-NVFP4",
        "--local-dir-use-symlinks", "False",
    ],
    "Gemma-4-31B-IT-NVFP4": [
        "huggingface-cli", "download",
        "nvidia/Gemma-4-31B-IT-NVFP4",
        "--local-dir", "./models/Gemma-4-31B-IT-NVFP4",
        "--local-dir-use-symlinks", "False",
    ],
}

# ── Global state ────────────────────────────────────────────────────


class LLMStatus(str, Enum):
    READY = "ready"
    SWITCHING = "switching"
    ERROR = "error"


_llm_state: dict[str, Any] = {
    "status": LLMStatus.READY,
    "current_model": None,       # populated on first query / startup
    "target_model": None,
    "error": None,
    "started_at": None,
    "completed_at": None,
}


def get_llm_state() -> dict[str, Any]:
    """Return a copy of the current LLM switching state."""
    state = {**_llm_state}
    if state["current_model"] is None:
        state["current_model"] = read_current_model_from_env()
    return state


# ── Public helpers ──────────────────────────────────────────────────

MODELS_DIR = get_project_root() / "models"


def list_available_models() -> list[str]:
    """Return the hardcoded list of allowed model names.

    Unlike the old implementation that scanned ``./models/``, this now
    returns the fixed set so the admin UI always shows all three models
    regardless of whether they're downloaded yet.
    """
    return sorted(ALLOWED_MODELS)


def read_current_model_from_env() -> str | None:
    """Read ``LLM_MODEL_DIR_NAME`` from the ``.env`` file."""
    env_path = get_project_root() / ".env"
    if not env_path.exists():
        return None
    text = env_path.read_text(encoding="utf-8")
    m = re.search(r"^[ \t]*LLM_MODEL_DIR_NAME[ \t]*=[ \t]*(.+?)[ \t]*$", text, re.MULTILINE)
    return m.group(1).strip() if m else None


# ── .env updater ────────────────────────────────────────────────────

def _update_env_file(model_dir_name: str) -> None:
    """Overwrite ``LLM_MODEL_DIR_NAME`` in the root ``.env`` file."""
    env_path = get_project_root() / ".env"
    if not env_path.exists():
        env_path.touch()
    text = env_path.read_text(encoding="utf-8")
    new_text, count = re.subn(
        r"^[ \t]*LLM_MODEL_DIR_NAME[ \t]*=.*$",
        f"LLM_MODEL_DIR_NAME={model_dir_name}",
        text,
        flags=re.MULTILINE,
    )
    if count == 0:
        # Key doesn't exist yet — append it
        if text and not text.endswith("\n"):
            text += "\n"
        new_text = text + f"LLM_MODEL_DIR_NAME={model_dir_name}\n"
    env_path.write_text(new_text, encoding="utf-8")
    
    # CRITICAL: Update os.environ so subprocess.run passes the new variable to Docker!
    import os
    os.environ["LLM_MODEL_DIR_NAME"] = model_dir_name
    
    logger.info("Updated .env: LLM_MODEL_DIR_NAME=%s", model_dir_name)




# ── Docker Compose YAML rewriter ───────────────────────────────────

# Template for Gemma (different image, no depends_on, different args)
_GEMMA_VLLM_SERVICE = """\
  vllm-llm:
    image: vllm/vllm-openai:gemma4-cu130
    container_name: vllm-llm
    restart: unless-stopped
    ports:
      - "${VLLM_PORT:-8001}:8001"
    shm_size: "64gb"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [ gpu ]
    volumes:
      - ./hf_cache:/root/.cache/huggingface
      - ./models:/models
    environment:
      - VLLM_LOGGING_LEVEL=INFO
      - VLLM_ATTENTION_BACKEND=FLASH_ATTN
    command: >
      --model /models/${LLM_MODEL_DIR_NAME}
      --served-model-name ${LLM_MODEL_DIR_NAME}
      --port 8001
      --host 0.0.0.0
      --max-model-len 16384
      --gpu-memory-utilization 0.40
      --max-num-seqs 128
      --tensor-parallel-size 1"""

# Template for Qwen / Phi (standard NVIDIA image, depends_on reranker)
_STANDARD_VLLM_SERVICE = """\
  vllm-llm:
    image: nvcr.io/nvidia/vllm:26.04-py3
    container_name: vllm-llm
    restart: unless-stopped
    ports:
      - "${VLLM_PORT:-8001}:8001"
    shm_size: "64gb"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [ gpu ]
    volumes:
      - ./hf_cache:/root/.cache/huggingface
      - ./models:/models
    environment:
      - VLLM_LOGGING_LEVEL=INFO
      - VLLM_ATTENTION_BACKEND=FLASH_ATTN
    depends_on:
      vllm-reranker:
        condition: service_healthy
    command: >
      vllm serve /models/${LLM_MODEL_DIR_NAME}  
      --served-model-name ${LLM_MODEL_DIR_NAME}  
      --port 8001 
      --trust-remote-code  
      --max-model-len 16384  
      --gpu-memory-utilization 0.40  
      --max-num-seqs 128  
      --tensor-parallel-size 1"""


def _rewrite_docker_compose(model_dir_name: str) -> None:
    """Rewrite the ``vllm-llm`` service block in ``docker-compose.yml``.

    - Gemma  → uses ``vllm/vllm-openai:gemma4-cu130``, no ``depends_on``,
               ``--host 0.0.0.0``, no ``vllm serve`` prefix.
    - Others → uses ``nvcr.io/nvidia/vllm:26.04-py3``, ``depends_on``
               reranker, ``vllm serve`` prefix, ``--trust-remote-code``.
    """
    compose_path = get_project_root() / "docker-compose.yml"
    content = compose_path.read_text(encoding="utf-8")

    is_gemma = model_dir_name == "Gemma-4-31B-IT-NVFP4"
    new_service = _GEMMA_VLLM_SERVICE if is_gemma else _STANDARD_VLLM_SERVICE

    # Use regex to replace the entire vllm-llm service block.
    # The block starts at "  vllm-llm:" and ends just before the next
    # top-level service definition (2-space indented name + colon) or
    # before "networks:" / end-of-file.
    pattern = re.compile(
        r"^  vllm-llm:.*?(?=\n  \S|\nnetworks:|\Z)",
        re.MULTILINE | re.DOTALL,
    )

    new_content, count = pattern.subn(new_service, content)
    if count == 0:
        raise RuntimeError(
            "Could not find 'vllm-llm' service block in docker-compose.yml"
        )

    compose_path.write_text(new_content, encoding="utf-8")
    logger.info(
        "Rewrote docker-compose.yml vllm-llm service for %s (gemma=%s)",
        model_dir_name, is_gemma,
    )


# ── Model downloader ───────────────────────────────────────────────

def _ensure_model_downloaded(model_dir_name: str) -> None:
    """Download the model from HuggingFace if it doesn't exist on disk.

    Uses ``huggingface-cli download`` with the exact arguments specified
    per model.  This is a blocking call that may take a long time for
    large models (e.g. Gemma 31B).
    """
    model_path = MODELS_DIR / model_dir_name
    if model_path.exists() and any(model_path.iterdir()):
        logger.info("Model directory already exists: %s", model_path)
        return

    cmd = _DOWNLOAD_COMMANDS.get(model_dir_name)
    if cmd is None:
        raise ValueError(f"No download command configured for model: {model_dir_name}")

    logger.info("Downloading model '%s' via huggingface-cli ...", model_dir_name)
    result = subprocess.run(
        cmd,
        cwd=str(get_project_root()),
        capture_output=True,
        text=True,
        # No timeout — large models can take hours on slow connections
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"huggingface-cli download failed (rc={result.returncode}): "
            f"{result.stderr}"
        )
    logger.info("Download complete for '%s'", model_dir_name)


# ── Docker helpers ──────────────────────────────────────────────────

def _docker_compose_cmd() -> list[str]:
    """Return the base docker compose command tokens."""
    return ["docker", "compose"]


def _run_docker(*args: str) -> subprocess.CompletedProcess:
    """Run a docker compose sub-command from the project root."""
    cmd = _docker_compose_cmd() + list(args)
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=str(get_project_root()),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        logger.error("Docker command failed (rc=%d): %s", result.returncode, result.stderr)
    else:
        logger.info("Docker command succeeded: %s", result.stdout.strip())
    return result


# ── Health polling ──────────────────────────────────────────────────

_VLLM_HEALTH_URL = "http://localhost:8001/v1/models"
_POLL_INTERVAL = 5        # seconds
_POLL_TIMEOUT = 600       # 10 minutes max


def _wait_for_vllm_ready(timeout: int = _POLL_TIMEOUT) -> bool:
    """Block until vLLM's ``/v1/models`` endpoint returns 200."""
    deadline = time.time() + timeout
    logger.info("Polling %s for readiness (timeout=%ds)…", _VLLM_HEALTH_URL, timeout)
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=5) as client:
                resp = client.get(_VLLM_HEALTH_URL)
                if resp.status_code == 200:
                    logger.info("vLLM is ready! Response: %s", resp.text[:200])
                    return True
        except Exception:
            pass
        time.sleep(_POLL_INTERVAL)
    logger.error("vLLM did not become ready within %ds.", timeout)
    return False


# ── Main switch routine (runs in BackgroundTask) ────────────────────

def switch_llm_model(target_model: str) -> None:
    """Full model-switch pipeline.  This is meant to be called inside a
    ``BackgroundTask`` so the API endpoint can return immediately.

    Includes **automatic rollback**: if any step fails (OOM, timeout,
    docker error), the ``.env`` and ``docker-compose.yml`` are restored
    to the previous working state and the container is restarted,
    preventing the application from going permanently offline.

    Steps:
      0. Save current model + docker-compose.yml as backup.
      1. Download model if not already on disk.
      2. Update ``.env`` with the new model name.
      3. Rewrite ``docker-compose.yml`` for the target model.
      4. ``docker compose stop vllm-llm`` (frees VRAM).
      5. ``docker compose up -d vllm-llm`` (starts new model).
      6. Poll health endpoint until ready (10-min timeout).
      7. Update ``config.json`` so the app routes requests correctly.
    """
    global _llm_state

    # ── 0. Snapshot the currently running model for rollback ─────
    backup_model = read_current_model_from_env()
    compose_path = get_project_root() / "docker-compose.yml"
    backup_compose = compose_path.read_text(encoding="utf-8")

    logger.info(
        "Starting model switch: %s → %s (backup=%s)",
        backup_model, target_model, backup_model,
    )

    _llm_state.update(
        status=LLMStatus.SWITCHING,
        target_model=target_model,
        error=None,
        started_at=time.time(),
        completed_at=None,
    )

    try:
        # 1. Download the model if it's not on disk yet
        _ensure_model_downloaded(target_model)

        # 2. Update .env
        _update_env_file(target_model)

        # 3. Rewrite docker-compose.yml for the target model
        _rewrite_docker_compose(target_model)

        # 4. Stop the current vllm-llm container (frees VRAM)
        try:
            result = _run_docker("stop", "vllm-llm")
            if result.returncode != 0:
                raise RuntimeError(f"docker compose stop failed: {result.stderr}")

            # 5. Recreate & start with new model
            result = _run_docker("up", "-d", "vllm-llm")
            if result.returncode != 0:
                raise RuntimeError(f"docker compose up failed: {result.stderr}")
        except Exception as docker_exc:
            raise RuntimeError(f"Docker compose error: {docker_exc}")

        # 6. Poll until the model is ready (10-minute timeout)
        if not _wait_for_vllm_ready():
            raise TimeoutError(
                f"vLLM did not become ready after switching to {target_model}"
            )

        # 7. Update config.json so the app sends the correct model name
        update_config({"models": {"default_llm": target_model}}, persist=True)
        logger.info(
            "Model switch complete: now serving %s", target_model
        )

        _llm_state.update(
            status=LLMStatus.READY,
            current_model=target_model,
            target_model=None,
            completed_at=time.time(),
        )

    except Exception as exc:
        logger.exception("Model switch to '%s' failed — initiating rollback", target_model)

        # ── Rollback: restore the previous working model ────────
        _perform_rollback(
            backup_model,
            backup_compose=backup_compose,
            original_error=exc,
        )


def _perform_rollback(
    backup_model: str | None,
    *,
    backup_compose: str,
    original_error: Exception,
) -> None:
    """Best-effort rollback to *backup_model* after a failed switch.

    Restores the ``.env``, ``docker-compose.yml``, restarts the
    vllm-llm container, and updates ``_llm_state`` regardless of
    whether the rollback itself succeeds.
    """
    global _llm_state
    error_detail = str(original_error)

    if not backup_model:
        logger.error("No backup model recorded — cannot rollback")
        _llm_state.update(
            status=LLMStatus.ERROR,
            error=f"Switch failed and no backup model to rollback to: {error_detail}",
            completed_at=time.time(),
        )
        return

    logger.info("Rolling back to previous model: %s", backup_model)

    try:
        # 1. Restore .env
        _update_env_file(backup_model)

        # 2. Restore docker-compose.yml from backup
        compose_path = get_project_root() / "docker-compose.yml"
        compose_path.write_text(backup_compose, encoding="utf-8")
        logger.info("Restored docker-compose.yml from backup")

        # 3. Stop whatever is (or isn't) running
        try:
            _run_docker("stop", "vllm-llm")
        except Exception as e:
            logger.warning("Rollback stop container failed, continuing: %s", e)

        # 4. Bring the old model back up
        try:
            result = _run_docker("up", "-d", "vllm-llm")
            if result.returncode != 0:
                raise RuntimeError(f"Rollback docker up failed: {result.stderr}")
        except Exception as e:
            raise RuntimeError(f"Rollback docker up failed: {e}")

        # 5. Wait for the old model to come back online
        if not _wait_for_vllm_ready():
            raise TimeoutError(
                f"Rollback model '{backup_model}' did not become ready"
            )

        # Rollback succeeded — old model is serving again
        update_config({"models": {"default_llm": backup_model}}, persist=True)
        logger.info("Rollback successful — restored to %s", backup_model)
        _llm_state.update(
            status=LLMStatus.ERROR,
            current_model=backup_model,
            target_model=None,
            error=f"Switch failed (rolled back to {backup_model}): {error_detail}",
            completed_at=time.time(),
        )

    except Exception as rollback_exc:
        logger.exception("Rollback itself also failed")
        _llm_state.update(
            status=LLMStatus.ERROR,
            error=(
                f"Switch failed: {error_detail} | "
                f"Rollback also failed: {rollback_exc}"
            ),
            completed_at=time.time(),
        )

def sync_llm_state_on_startup() -> None:
    """
    Source of Truth Failsafe:
    Queries vLLM directly on startup. If online, syncs the .env to match the VRAM model.
    If offline, triggers a background repair to rebuild and restart the target model.
    """
    import threading

    global _llm_state
    
    # We use whatever is in .env as our target "default" if we need to repair.
    target_model = read_current_model_from_env()
    if not target_model or target_model not in ALLOWED_MODELS:
        target_model = "Qwen2.5-VL-7B-Instruct-NVFP4"

    # Temporarily set to SWITCHING so users see Maintenance Mode while we verify
    # or repair the LLM state
    _llm_state.update(
        status=LLMStatus.SWITCHING,
        target_model=target_model,
        error=None,
        started_at=time.time(),
        completed_at=None,
    )

    logger.info("Running startup LLM state diagnostic...")
    
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(_VLLM_HEALTH_URL)
            resp.raise_for_status()
            data = resp.json()
            if "data" in data and len(data["data"]) > 0:
                true_model = data["data"][0]["id"]
            else:
                raise ValueError("Unexpected /v1/models response format")

        if true_model != target_model:
            logger.warning(
                "LLM state mismatch! .env expects '%s' but VRAM has '%s'. "
                "VRAM is the source of truth. Overwriting .env...",
                target_model, true_model
            )
            _update_env_file(true_model)
            update_config({"models": {"default_llm": true_model}}, persist=True)
            target_model = true_model

        logger.info("LLM state synced. Current model in VRAM: %s", target_model)
        _llm_state.update(
            status=LLMStatus.READY,
            current_model=target_model,
            target_model=None,
            error=None,
            completed_at=time.time()
        )

    except Exception as e:
        logger.error(
            "vLLM offline or error during startup diagnostic (%s). "
            "Forcing repair/rebuild to target model: %s",
            e, target_model
        )
        
        # Trigger hidden background repair using the existing robust switch routine
        threading.Thread(target=switch_llm_model, args=(target_model,), daemon=True).start()

def get_active_vllm_model_name() -> str | None:
    """Queries the running vLLM container to get the true active model ID."""
    try:
        with httpx.Client(timeout=2) as client:
            resp = client.get(_VLLM_HEALTH_URL)
            if resp.status_code == 200:
                data = resp.json()
                if "data" in data and len(data["data"]) > 0:
                    return data["data"][0]["id"]
    except Exception as e:
        logger.warning("Failed to query vLLM for active model: %s", e)
    return None

