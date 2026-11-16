"""File validation: name, extension, size, content signature and non-emptiness."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

from copilot.config import Settings
from copilot.ingestion.models import FileType


class IngestionError(Exception):
    """Base class for expected, user-facing ingestion problems."""


class ValidationFailed(IngestionError):
    """The uploaded file or its metadata is not acceptable."""


def safe_file_name(filename: str) -> str:
    """Return a clean base name or raise :class:`ValidationFailed`."""
    name = Path((filename or "").replace("\\", "/")).name.strip()
    if not name or name in {".", ".."}:
        raise ValidationFailed("File name is empty.")
    if len(name) > 255:
        raise ValidationFailed("File name is too long (max 255 characters).")
    if any(ord(ch) < 32 for ch in name):
        raise ValidationFailed("File name contains control characters.")
    return name


def validate_upload(filename: str, data: bytes, settings: Settings) -> FileType:
    """Validate extension, size and content signature; return the file type.

    The "MIME" check is a content-signature sniff (magic bytes), which is more
    reliable than trusting a client-supplied Content-Type header.
    """
    name = safe_file_name(filename)
    ext = Path(name).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise ValidationFailed(
            f"Unsupported file type '{ext or 'none'}'. Allowed: {', '.join(settings.allowed_extensions)}."
        )
    if not data:
        raise ValidationFailed("File is empty.")
    if len(data) > settings.max_upload_bytes:
        raise ValidationFailed(f"File exceeds the {settings.max_upload_mb} MB upload limit.")

    if ext == ".pdf":
        if not data.lstrip()[:5] == b"%PDF-":
            raise ValidationFailed("File content is not a valid PDF.")
        return "pdf"
    if ext == ".docx":
        if not data.startswith(b"PK"):
            raise ValidationFailed("File content is not a valid DOCX.")
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                if "word/document.xml" not in zf.namelist():
                    raise ValidationFailed("File content is not a valid DOCX (missing word/document.xml).")
        except zipfile.BadZipFile as exc:
            raise ValidationFailed("File content is not a valid DOCX (corrupt archive).") from exc
        return "docx"
    # .txt
    if b"\x00" in data[:8192]:
        raise ValidationFailed("File looks binary, not plain text.")
    return "txt"
