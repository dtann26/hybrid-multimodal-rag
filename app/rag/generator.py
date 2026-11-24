"""
LLM generator — produces chat completions via vLLM's OpenAI-compatible HTTP API.

The vLLM server runs as a separate container (see docker-compose.yml).
This module calls it using the standard OpenAI SDK, which means:
  - No model is loaded in-process.
  - Streaming is real token-by-token (not a chunked simulation).
  - Both text and vision-language models are supported through the same API
    (Qwen2.5-VL is a VLM served via the single LLM endpoint).

Configure the server URL via ``models.llm_api_endpoint`` in config.json
(default: http://localhost:8001/v1).
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any

from app.core.config import get

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "http://localhost:8001/v1"
_DEFAULT_API_KEY = "not-needed"


# ── Client helper ─────────────────────────────────────────────────

def _get_client(api_endpoint: str | None = None, *, streaming: bool = False):
    """Return an OpenAI client pointed at the vLLM server.

    *streaming*: when True, disables the read timeout so long token streams
    are never cut off.  Connection timeout (10s) still applies in both modes.
    """
    from openai import OpenAI
    import httpx

    base_url = api_endpoint or get("models.llm_api_endpoint", _DEFAULT_ENDPOINT)

    if streaming:
        # Streaming: no read timeout to allow long responses.
        # Connection timeout still applies.
        timeout = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)
    else:
        # Blocking: 5-minute read timeout for planner / non-streaming calls.
        timeout = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)

    return OpenAI(base_url=base_url, api_key=_DEFAULT_API_KEY, timeout=timeout)


def _to_openai_messages(messages: list[dict]) -> list[dict]:
    """Convert internal message format to the OpenAI API wire format.

    Translates ``{"type": "image", "image": "<data-uri>"}`` parts
    to ``{"type": "image_url", "image_url": {"url": "<data-uri>"}}``
    so vLLM's multimodal endpoint accepts them correctly.
    """
    result = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, list):
            parts = []
            for part in content:
                if part.get("type") == "text":
                    parts.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "image":
                    parts.append({"type": "image_url", "image_url": {"url": part["image"]}})
            result.append({"role": role, "content": parts})
        else:
            result.append({"role": role, "content": content})
    return result


# ── Generation ───────────────────────────────────────────────────

def generate(
    messages: list[dict[str, str]],
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    model_path: str | None = None,
    api_endpoint: str | None = None,
) -> str:
    """Generate a complete response (non-streaming) via the vLLM HTTP API.

    *messages* follows the OpenAI chat format::

        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

    Multimodal messages (with image content) are automatically converted to
    the ``image_url`` format expected by vLLM's vision endpoint.

    ``model_path`` is the ``--served-model-name`` value passed to vLLM
    (e.g. ``"Qwen2.5-VL-7B-Instruct-NVFP4"``).  Falls back to
    ``models.default_llm`` in config.
    """
    model_name = model_path or get("models.default_llm", "Qwen2.5-VL-7B-Instruct-NVFP4")
    client = _get_client(api_endpoint)

    response = client.chat.completions.create(
        model=model_name,
        messages=_to_openai_messages(messages),
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
        stream=False,
    )
    return response.choices[0].message.content or ""


def generate_stream(
    messages: list[dict[str, str]],
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    model_path: str | None = None,
    api_endpoint: str | None = None,
) -> Generator[str, None, None]:
    """Generate response tokens via real streaming from the vLLM HTTP API.

    Yields each token string as it arrives from the server.
    """
    model_name = model_path or get("models.default_llm", "Qwen2.5-VL-7B-Instruct-NVFP4")
    client = _get_client(api_endpoint, streaming=True)

    stream = client.chat.completions.create(
        model=model_name,
        messages=_to_openai_messages(messages),
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ── Token counting ────────────────────────────────────────────────

# Budget for chat history tokens.  The system prompt + context + user
# message consume the rest of the context window; history gets whatever
# is left up to this cap.
_MAX_HISTORY_TOKENS = 2048


def _estimate_tokens(text: str) -> int:
    """Accurate token count using tiktoken's cl100k_base encoding."""
    if not text:
        return 0
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Fallback to character estimate if tiktoken is not available
        return len(text) // 4


def _truncate_history(
    history: list[dict[str, Any]],
    max_tokens: int = _MAX_HISTORY_TOKENS,
) -> list[dict[str, Any]]:
    """Keep the most recent messages that fit within *max_tokens*.

    Drops oldest messages first.  Always preserves at least the most
    recent message if it fits.
    """
    if not history:
        return []

    total = 0
    kept: list[dict[str, Any]] = []
    # Walk backwards (newest first) and accumulate
    for msg in reversed(history):
        msg_tokens = msg.get("token_count")
        if msg_tokens is None:
            msg_tokens = _estimate_tokens(msg.get("content", ""))
            
        if total + msg_tokens > max_tokens and kept:
            break  # budget exhausted — stop adding older messages
        total += msg_tokens
        kept.append(msg)

    kept.reverse()
    if len(kept) < len(history):
        logger.debug(
            "History truncated from %d to %d messages to fit token budget.",
            len(history), len(kept),
        )
    return kept


# ── RAG message builder ───────────────────────────────────────────

def build_rag_messages(
    query: str,
    context_chunks: list[dict[str, Any]],
    system_prompt: str,
    chat_history: list[dict[str, str]] | None = None,
    images: list[str] | None = None,
) -> list[dict]:
    """Build the chat messages list for a RAG query.

    Inserts retrieved context into the system prompt with document titles.
    Format: [Document Title: {title}]\n{chunk_content}
    
    Falls back to [Document Title: Unknown] if title is missing or empty.
    
    If *images* is provided (list of base64 data URIs), the user message
    is formatted as a multimodal content list for VLM.

    Chat history is token-aware truncated to stay within a ~2048 token
    budget, dropping the oldest messages first.
    """
    context_parts: list[str] = []
    for i, chunk in enumerate(context_chunks, 1):
        # Extract document title with fallback
        doc_title = chunk.get("document_title") or "Unknown"
        if not doc_title or not str(doc_title).strip():
            doc_title = "Unknown"
        
        # Build source info for location references
        source_info = ""
        if chunk.get("source_type") == "youtube" and chunk.get("start_time") is not None:
            source_info = f" [Video @ {chunk['start_time']:.0f}s]"
        elif chunk.get("page_start") and chunk["page_start"] > 0:
            if chunk.get("page_end") and chunk["page_end"] != chunk["page_start"]:
                source_info = f" [Pages {chunk['page_start']}-{chunk['page_end']}]"
            else:
                source_info = f" [Page {chunk['page_start']}]"
        
        chunk_content = chunk.get("content", "")
        titled_chunk = f"[Document Title: {doc_title}]\n{chunk_content}{source_info}"
        context_parts.append(f"[{i}] {titled_chunk}")

    context_text = "\n\n".join(context_parts)

    full_system = (
        f"{system_prompt}\n\n"
        f"--- Retrieved Context ---\n"
        f"{context_text}\n"
        f"--- End Retrieved Context ---\n\n"
        f"INSTRUCTIONS:\n"
        f"1. The numbered sources above — [1], [2], [3], etc. — are your ONLY allowed sources.\n"
        f"2. Answer using ONLY the information in those sources. Do not use external knowledge.\n"
        f"3. CITATION RULE (MANDATORY): Every factual claim MUST end with the exact reference number(s) of the source(s) that support it, in the format [N].\n"
        f"   - Correct: 'The sermon emphasized faith and victory [1].'\n"
        f"   - Correct: 'Prayer and doctrine are central themes [1][3].'\n"
        f"   - INCORRECT: 'The sermon emphasized faith and victory.' (missing citation)\n"
        f"4. If the answer is not found in the sources, say 'I do not have enough information to answer that.'\n"
        f"5. Failure to include [N] citations makes your response invalid.\n\n"
        f"Now answer the user's question."
    )

    messages: list[dict] = [{"role": "system", "content": full_system}]

    if chat_history:
        trimmed = _truncate_history(chat_history[-10:], _MAX_HISTORY_TOKENS)
        for msg in trimmed:
            messages.append({"role": msg["role"], "content": msg["content"]})

    if images:
        user_content: list[dict[str, str]] = []
        for img in images:
            user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": query})
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": query})

    return messages

### For comparison, also build a simpler message list without the RAG instructions ###

# ───────────────────────────────────────────Start───────────────────────────────────────────
def build_rag_messages_base(
    query: str,
    system_prompt: str,
    chat_history: list[dict[str, str]] | None = None,
) -> list[dict]:
    """Build base chat messages without retrieved context.

    Used for the initial planner call before retrieval, and for non-RAG queries.
    """
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    if chat_history:
        trimmed = _truncate_history(chat_history[-10:], _MAX_HISTORY_TOKENS)
        for msg in trimmed:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": query})
    return messages

def build_rag_messages_no_rerank(
    query: str,
    system_prompt: str,
    chat_history: list[dict[str, str]] | None = None,
) -> list[dict]:
    """Build RAG messages without reranking.

    Used when reranking is disabled.
    """
    return build_rag_messages_base(query, system_prompt, chat_history)

# ───────────────────────────────────────────End────────────────────────────────────────────