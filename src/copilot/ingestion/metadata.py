"""Document metadata: header parsing and resolution of overrides/defaults.

A document may begin with a disclaimer line followed by ``Key: Value`` lines
(Title, Category, Department, Access-Level, Version, Last-Reviewed, Status). Explicit
arguments (e.g. from the admin upload form) always win over header values.
"""
from __future__ import annotations

import re
from pathlib import Path

from copilot.ingestion.models import DocumentMetadata, LoadedDocument, PageText
from copilot.ingestion.validators import ValidationFailed

DISCLAIMER_PREFIX = "SYNTHETIC DEMO DOCUMENT"
VALID_ACCESS_LEVELS = ("public", "internal", "restricted")
_KNOWN_KEYS = {"title", "category", "department", "access-level", "version", "last-reviewed", "status"}
# "Status: current" or "Status: superseded [by <document>]". Any other value is not a status line.
_STATUS_VALUE = re.compile(r"^(current|superseded)(?:\s+by\s+(.+))?$", re.IGNORECASE)
_HEADER_LINE = re.compile(r"^([A-Za-z][A-Za-z _-]{1,30}):\s*(.+)$")


def split_header(text: str) -> tuple[dict[str, str], str]:
    """Split a leading metadata block from the body.

    Returns ``(header_dict, remaining_text)``. Keys are lower-cased with hyphens.
    If there is no header, the text is returned unchanged.
    """
    lines = text.split("\n")
    header: dict[str, str] = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            if header:
                i += 1
                break
            i += 1
            continue
        if line.upper().startswith(DISCLAIMER_PREFIX):
            i += 1
            continue
        match = _HEADER_LINE.match(line)
        key = match.group(1).strip().lower().replace(" ", "-").replace("_", "-") if match else ""
        if match and key == "status" and not _STATUS_VALUE.match(match.group(2).strip()):
            break  # not a status line (for example body text that happens to start with "Status:")
        if match and key in _KNOWN_KEYS:
            header[key] = match.group(2).strip()
            i += 1
            continue
        break
    if not header:
        return {}, text
    return header, "\n".join(lines[i:])


def parse_status(value: str | None) -> tuple[str, str | None]:
    """Return ``(status, superseded_by)`` from a header value; a missing value means ``current``."""
    match = _STATUS_VALUE.match((value or "").strip())
    if not match:
        return "current", None
    status = match.group(1).lower()
    by = (match.group(2) or "").strip() or None
    return status, (by if status == "superseded" else None)


def _humanize(file_name: str) -> str:
    return Path(file_name).stem.replace("_", " ").replace("-", " ").strip()


def extract_metadata(
    loaded: LoadedDocument,
    *,
    category: str | None = None,
    department: str | None = None,
    access_level: str | None = None,
) -> tuple[DocumentMetadata, LoadedDocument]:
    """Resolve metadata and return it with the document stripped of its header block."""
    header: dict[str, str] = {}
    pages = list(loaded.pages)
    if pages:
        header, body = split_header(pages[0].text)
        pages[0] = PageText(page_number=pages[0].page_number, text=body)

    level = (access_level or header.get("access-level") or "internal").strip().lower()
    if level not in VALID_ACCESS_LEVELS:
        raise ValidationFailed(
            f"Invalid access level '{level}'. Allowed: {', '.join(VALID_ACCESS_LEVELS)}."
        )
    status, superseded_by = parse_status(header.get("status"))
    metadata = DocumentMetadata(
        title=header.get("title") or _humanize(loaded.file_name),
        category=(category or header.get("category") or "general").strip().lower(),
        department=(department or header.get("department") or "IT").strip(),
        access_level=level,  # type: ignore[arg-type]
        last_reviewed=header.get("last-reviewed"),
        source_version=header.get("version"),
        status=status,  # type: ignore[arg-type]
        superseded_by=superseded_by,
    )
    return metadata, loaded.model_copy(update={"pages": pages})
