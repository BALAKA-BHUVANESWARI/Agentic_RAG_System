"""Query sanitisation for PUBLIC web search (pure functions, no network).

Two severities:
* ``block``: secrets/credentials. External search is refused entirely and the text is
  redacted before being stored in audit logs.
* ``strip``: identifiers (emails, IPs, hostnames, ticket IDs, internal URLs, paths,
  usernames, employee IDs) and configured private terms. These are removed from the
  query; the remaining generic words may still be searched.

This is a defence-in-depth layer: it reduces leakage risk but cannot guarantee that
free text never contains confidential information.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# (kind, severity, pattern). Applied in order; matched text is blanked so later patterns
# do not re-match inside it (e.g. an email is not re-reported as a hostname).
_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    ("private_key", "block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("api_key", "block", re.compile(r"\b(?:gsk_|pcsk_|tvly-|sk-|ghp_|xox[abp]-|AKIA)[A-Za-z0-9_\-]{8,}")),
    ("credential", "block", re.compile(r"(?i)\b(?:password|passwd|pwd|passcode|secret|token|api[_-]?key)\b\s*(?:is|=|:)\s*\S+")),
    ("bearer_token", "block", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{10,}")),
    ("long_secret", "block", re.compile(r"\b[A-Za-z0-9+/_\-]{40,}\b")),
    ("email", "strip", re.compile(r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+")),
    ("url", "strip", re.compile(r"https?://\S+")),
    ("ip_address", "strip", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("hostname", "strip", re.compile(r"\b[\w\-]+(?:\.[\w\-]+)*\.(?:example|local|internal|corp|lan|intranet|test)\b", re.I)),
    ("ticket_id", "strip", re.compile(r"\b(?:INC|TKT|REQ|CHG|SR)[-_ ]?\d{4,}\b|\b[A-Z]{2,6}-\d{3,}\b")),
    ("windows_path", "strip", re.compile(r"\b[A-Za-z]:\\\S+|\\\\[\w.\-]+\\\S+")),
    ("domain_user", "strip", re.compile(r"\b[A-Za-z0-9_\-]+\\[A-Za-z0-9_.\-]+\b")),
    ("employee_id", "strip", re.compile(r"\b(?:EMP|E)[-_]?\d{4,}\b")),
]

MIN_QUERY_WORDS = 3
MIN_QUERY_CHARS = 12


@dataclass(frozen=True)
class Finding:
    """One sensitive span found in text."""

    kind: str
    severity: str  # "block" | "strip"


@dataclass
class SanitizedQuery:
    """Result of sanitising text for public search."""

    query: str
    removed: list[str] = field(default_factory=list)
    blocked: bool = False
    blocked_reason: str | None = None
    usable: bool = False


def find_sensitive(text: str) -> list[Finding]:
    """List sensitive findings (kinds and severities) without returning the matched text."""
    findings: list[Finding] = []
    work = text
    for kind, severity, pattern in _PATTERNS:
        if pattern.search(work):
            findings.append(Finding(kind, severity))
            work = pattern.sub(" ", work)
    return findings


def has_blocking_secret(text: str) -> bool:
    """True if the text contains something that looks like a credential or secret."""
    return any(f.severity == "block" for f in find_sensitive(text))


def redact_sensitive(text: str) -> str:
    """Replace secret-like spans (block severity) with ``[REDACTED]`` (for audit logs)."""
    for _kind, severity, pattern in _PATTERNS:
        if severity == "block":
            text = pattern.sub("[REDACTED]", text)
    return text


def _truncate_at_word(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    return cut.rsplit(" ", 1)[0] if " " in cut else cut


def sanitize_query(text: str, private_terms: list[str] | None = None, max_chars: int = 200) -> SanitizedQuery:
    """Make ``text`` safe(r) for a public search engine.

    Blocks on secrets; otherwise strips identifiers and private terms, collapses
    whitespace, truncates, and reports whether enough generic content remains.
    """
    findings = find_sensitive(text)
    blocking = [f for f in findings if f.severity == "block"]
    if blocking:
        kinds = ", ".join(sorted({f.kind for f in blocking}))
        return SanitizedQuery(query="", removed=[f.kind for f in findings], blocked=True,
                              blocked_reason=f"possible secret or credential in the question ({kinds})")
    work = text
    removed: list[str] = []
    for kind, severity, pattern in _PATTERNS:
        if severity == "strip" and pattern.search(work):
            removed.append(kind)
            work = pattern.sub(" ", work)
    for term in sorted(private_terms or [], key=len, reverse=True):
        pat = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
        if pat.search(work):
            removed.append("private_term")
            work = pat.sub(" ", work)
    work = re.sub(r"[\(\[\{]\s*[\)\]\}]", " ", work)  # brackets emptied by stripping
    work = re.sub(r"\s+", " ", work).strip(" \t\r\n.,;:-")
    work = _truncate_at_word(work, max_chars)
    usable = len(work.split()) >= MIN_QUERY_WORDS and len(work) >= MIN_QUERY_CHARS
    return SanitizedQuery(query=work, removed=removed, usable=usable)
