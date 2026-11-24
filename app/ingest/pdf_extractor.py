"""
PDF extractor — extract text and images from PDF files.

Uses PyMuPDF (fitz) for image extraction and MarkItDown for text.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_pdf_text(file_path: str | Path) -> str:
    """Extract text from a PDF using MarkItDown (which handles PDFs)."""
    from app.ingest.markdown_converter import convert_to_markdown
    return convert_to_markdown(file_path)


def extract_pdf_images(file_path: str | Path, output_dir: str | Path) -> list[str]:
    """Extract images from a PDF using PyMuPDF.

    Returns a list of saved image file paths.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(str(file_path))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_images: list[str] = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)

        for img_idx, img_info in enumerate(images):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image_name = f"page{page_num + 1}_img{img_idx + 1}.{image_ext}"
            image_path = output_dir / image_name

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            saved_images.append(str(image_path))

    doc.close()
    logger.info("Extracted %d images from '%s'", len(saved_images), file_path)
    return saved_images
