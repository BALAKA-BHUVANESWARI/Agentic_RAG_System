"""Text cleaning applied to extracted document text."""
from __future__ import annotations

import re
import unicodedata

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_INLINE_SPACES = re.compile(r"[ \t ]+")
_HYPHEN_BREAK = re.compile(r"(\w)-\n([a-z])")
_MANY_BLANKS = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize line endings and whitespace while keeping line structure.

    Newlines are preserved because headings and list items depend on them.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("﻿", "")
    text = _CONTROL_CHARS.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)  # re-join words hyphenated across lines
    text = _INLINE_SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MANY_BLANKS.sub("\n\n", text)
    return text.strip()
