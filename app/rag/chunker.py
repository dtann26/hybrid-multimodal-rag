"""
Text chunker — splits documents into overlapping segments for embedding.

Supports plain text splitting by token/character count as well as
time-based splitting for video transcripts (reuses the existing
30-second fixed-segment logic from src/scrapper_transcriber/fix_segments.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.config import get_rag_config


@dataclass
class Chunk:
    index: int
    content: str
    start_time: float | None = None
    end_time: float | None = None
    page_start: int = 0
    page_end: int = 0
    metadata: dict = field(default_factory=dict)


def _split_by_markdown_headers(text: str) -> list[tuple[str, str]]:
    """Split *text* into ``(header, body)`` pairs on Markdown headers.

    Headers at levels 1–3 (``#`` through ``###``) are recognised.
    Text before the first header gets an empty-string header.
    """
    # Match lines that start with 1-3 '#' followed by a space
    header_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

    sections: list[tuple[str, str]] = []
    last_end = 0
    last_header = ""

    for match in header_pattern.finditer(text):
        # Flush everything before this header into the previous section
        body = text[last_end : match.start()].strip()
        if body or last_header:
            sections.append((last_header, body))
        last_header = match.group(0).strip()  # full header line
        last_end = match.end()

    # Trailing content after the last header
    tail = text[last_end:].strip()
    if tail or last_header:
        sections.append((last_header, tail))

    # If no headers were found at all, return the full text as one section
    if not sections:
        sections.append(("", text.strip()))

    return sections


def _recursive_char_split(
    text: str,
    size: int,
    overlap: int,
) -> list[str]:
    """Split *text* into overlapping character-based segments.

    Tries to break on paragraph boundaries first, then sentence
    boundaries (``". "``), then falls back to hard character cuts.
    """
    text = text.strip()
    if not text or len(text) <= size:
        return [text] if text else []

    segments: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        segment = text[start:end]

        if end < len(text):
            # Prefer paragraph break
            last_para = segment.rfind("\n\n")
            if last_para > size // 3:
                segment = segment[: last_para + 1]
                end = start + last_para + 1
            else:
                # Then sentence break
                last_period = segment.rfind(". ")
                last_newline = segment.rfind("\n")
                break_pos = max(last_period, last_newline)
                if break_pos > size // 2:
                    segment = segment[: break_pos + 1]
                    end = start + break_pos + 1

        segments.append(segment.strip())
        start = end - overlap
        if start >= len(text):
            break

    return segments


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Split *text* into overlapping chunks with Markdown-header awareness.

    The function first splits on Markdown headers (``#``, ``##``, ``###``)
    to preserve semantic section boundaries.  If a section exceeds
    *chunk_size*, it is recursively split using character-based sliding
    windows with paragraph/sentence boundary preference.

    Falls back to the global config values when ``chunk_size`` /
    ``chunk_overlap`` are not supplied.
    """
    cfg = get_rag_config()
    size = chunk_size or cfg["chunk_size"]
    overlap = chunk_overlap or cfg["chunk_overlap"]

    if not text or not text.strip():
        return []

    sections = _split_by_markdown_headers(text)

    chunks: list[Chunk] = []
    idx = 0

    for header, body in sections:
        # Prepend the header to the body so every chunk carries
        # its section context for better embedding quality.
        if header and body:
            section_text = f"{header}\n{body}"
        elif header:
            section_text = header
        else:
            section_text = body

        if not section_text.strip():
            continue

        if len(section_text) <= size:
            # Section fits in a single chunk
            chunks.append(Chunk(
                index=idx,
                content=section_text.strip(),
                metadata={"header": header} if header else {},
            ))
            idx += 1
        else:
            # Section is too large — recursively split it
            sub_segments = _recursive_char_split(section_text, size, overlap)
            for seg in sub_segments:
                if seg:
                    chunks.append(Chunk(
                        index=idx,
                        content=seg,
                        metadata={"header": header} if header else {},
                    ))
                    idx += 1

    return chunks


def chunk_pages(
    pages: list[dict],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Chunk text extracted page-by-page (e.g. from a PDF).

    *pages* is a list of ``{"page": 1, "text": "..."}`` dicts.
    Each resulting chunk records which page(s) it spans via
    ``page_start`` / ``page_end``.
    """
    if not pages:
        return []

    cfg = get_rag_config()
    size = chunk_size or cfg["chunk_size"]
    overlap = chunk_overlap or cfg["chunk_overlap"]

    chunks: list[Chunk] = []
    idx = 0

    for page_info in pages:
        page_num = page_info["page"]
        text = re.sub(r"\s+", " ", page_info["text"]).strip()
        if not text:
            continue

        # Split the page text into chunks
        start = 0
        while start < len(text):
            end = start + size
            segment = text[start:end]

            if end < len(text):
                last_period = segment.rfind(". ")
                last_newline = segment.rfind("\n")
                break_pos = max(last_period, last_newline)
                if break_pos > size // 2:
                    segment = segment[: break_pos + 1]
                    end = start + break_pos + 1

            chunks.append(Chunk(
                index=idx,
                content=segment.strip(),
                page_start=page_num,
                page_end=page_num,
            ))
            idx += 1
            start = end - overlap
            if start >= len(text):
                break

    return chunks


def chunk_transcript_segments(
    segments: list[dict],
    interval: float = 30.0,
) -> list[Chunk]:
    """Merge raw transcript segments into fixed-interval chunks.

    Each input segment has ``start``, ``end``, ``text`` keys (the
    format produced by Whisper / fix_segments.py).  Output chunks
    cover non-overlapping *interval*-second windows.
    """
    if not segments:
        return []

    # Determine the total duration
    max_end = max(s.get("end", 0) for s in segments)
    n_windows = int(max_end // interval) + 1

    buckets: list[list[str]] = [[] for _ in range(n_windows)]
    time_ranges: list[tuple[float, float]] = [
        (i * interval, (i + 1) * interval) for i in range(n_windows)
    ]

    for seg in segments:
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", seg_start)
        seg_text = seg.get("text", "").strip()
        if not seg_text:
            continue

        # Place segment into the bucket(s) it overlaps with
        for i, (win_start, win_end) in enumerate(time_ranges):
            if seg_start < win_end and seg_end > win_start:
                buckets[i].append(seg_text)

    chunks: list[Chunk] = []
    for i, texts in enumerate(buckets):
        if not texts:
            continue
        combined = " ".join(texts)
        ws, we = time_ranges[i]
        chunks.append(
            Chunk(index=i, content=combined, start_time=ws, end_time=we)
        )

    return chunks
