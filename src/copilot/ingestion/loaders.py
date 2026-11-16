"""Format-specific text extraction for PDF, DOCX and TXT."""
from __future__ import annotations

import io

from copilot.ingestion.models import FileType, LoadedDocument, PageText
from copilot.ingestion.validators import IngestionError


def _load_txt(file_name: str, data: bytes) -> LoadedDocument:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - latin-1 never fails, but keep a safe fallback
        text = data.decode("latin-1", errors="replace")
    return LoadedDocument(file_name=file_name, file_type="txt", pages=[PageText(text=text)])


def _load_docx(file_name: str, data: bytes) -> LoadedDocument:
    from docx import Document  # lazy: keeps import cost off the hot path
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        doc = Document(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001 - python-docx raises many types
        raise IngestionError(f"Could not read DOCX file: {type(exc).__name__}") from exc

    lines: list[str] = []
    for block in doc.iter_inner_content():
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                lines.append("")
                continue
            style = (block.style.name if block.style is not None else "") or ""
            if style == "Title":
                lines.append(f"# {text}")
            elif style.startswith("Heading"):
                lines.append(f"## {text}")
            else:
                lines.append(text)
        elif isinstance(block, Table):
            for row in block.rows:
                cells = [c.text.strip() for c in row.cells]
                lines.append(" | ".join(c for c in cells if c))
    return LoadedDocument(file_name=file_name, file_type="docx", pages=[PageText(text="\n".join(lines))])


def _load_pdf(file_name: str, data: bytes) -> LoadedDocument:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise IngestionError("PDF is password-protected.")
        pages = [
            PageText(page_number=i, text=page.extract_text() or "")
            for i, page in enumerate(reader.pages, start=1)
        ]
    except IngestionError:
        raise
    except Exception as exc:  # noqa: BLE001 - pypdf raises many types on malformed files
        raise IngestionError(f"Could not read PDF file: {type(exc).__name__}") from exc
    if not pages:
        raise IngestionError("PDF has no pages.")
    return LoadedDocument(file_name=file_name, file_type="pdf", pages=pages)


def load_document(file_name: str, file_type: FileType, data: bytes) -> LoadedDocument:
    """Extract raw text from ``data`` according to ``file_type``."""
    if file_type == "txt":
        return _load_txt(file_name, data)
    if file_type == "docx":
        return _load_docx(file_name, data)
    if file_type == "pdf":
        return _load_pdf(file_name, data)
    raise IngestionError(f"No loader for file type '{file_type}'.")
