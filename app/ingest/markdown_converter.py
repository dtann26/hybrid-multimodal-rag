"""
Markdown converter — converts various file formats to Markdown text.

Uses Microsoft's ``markitdown`` library for DOCX, PPTX, XLSX, HTML,
and other supported formats.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def convert_to_markdown(file_path: str | Path) -> str:
    """Convert a file to Markdown text using MarkItDown.

    Supported formats: DOCX, PPTX, XLSX, PDF, HTML, TXT, CSV, JSON, XML, etc.
    Returns the Markdown text content.
    """
    from markitdown import MarkItDown

    md = MarkItDown()
    result = md.convert(str(file_path))
    logger.info("Converted '%s' to markdown (%d chars)", file_path, len(result.text_content))
    return result.text_content


def convert_text_to_markdown(text: str, source_format: str = "txt") -> str:
    """Wrap plain text as Markdown (pass-through for already-markdown content)."""
    if source_format in ("md", "markdown"):
        return text
    return text
