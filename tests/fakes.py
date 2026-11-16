"""Test doubles and file builders. Fakes live in tests/ only, never in app code."""
from __future__ import annotations

import hashlib
import io
import math
import re
from typing import Any

from copilot.ingestion.models import Chunk

_STOPWORDS = {
    "a", "an", "the", "i", "do", "does", "how", "to", "from", "is", "are", "of", "in", "on", "my",
    "and", "or", "for", "with", "what", "can", "me", "it", "be", "if", "you", "your", "at", "by",
}


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder (no model download, no network)."""

    dimension = 512

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            if token in _STOPWORDS:
                continue
            stem = token[:6]  # crude stemming: connect/connecting/connection -> "connec"
            vec[int(hashlib.md5(stem.encode()).hexdigest(), 16) % self.dimension] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def _matches(metadata: dict[str, Any], flt: dict[str, Any]) -> bool:
    if "$and" in flt:
        return all(_matches(metadata, f) for f in flt["$and"])
    for key, cond in flt.items():
        value = metadata.get(key)
        if "$in" in cond and value not in cond["$in"]:
            return False
        if "$eq" in cond and value != cond["$eq"]:
            return False
    return True


class FakeIndex:
    """In-memory stand-in for a Pinecone index (same method names/shapes we use)."""

    def __init__(self, page_size: int = 2) -> None:
        self.data: dict[str, dict[str, tuple[list[float], dict[str, Any]]]] = {}
        self.page_size = page_size
        self.upsert_batches: list[int] = []
        self.query_calls = 0
        self.last_filter: dict[str, Any] | None = None

    def ids(self, namespace: str = "kb") -> list[str]:
        return sorted(self.data.get(namespace, {}))

    def upsert(self, vectors: list[dict[str, Any]], namespace: str | None = None) -> None:
        self.upsert_batches.append(len(vectors))
        ns = self.data.setdefault(namespace or "", {})
        for v in vectors:
            assert all(val is not None for val in v["metadata"].values()), "Pinecone rejects null metadata"
            ns[v["id"]] = (v["values"], v["metadata"])

    def query(self, vector, top_k, filter=None, include_metadata=True, namespace=None):  # noqa: A002
        self.query_calls += 1
        self.last_filter = filter
        scored = []
        for _id, (values, md) in self.data.get(namespace or "", {}).items():
            if filter and not _matches(md, filter):
                continue
            scored.append({"id": _id, "score": sum(a * b for a, b in zip(vector, values)), "metadata": md})
        scored.sort(key=lambda m: m["score"], reverse=True)
        return {"matches": scored[:top_k]}

    def list(self, prefix: str = "", namespace: str | None = None):
        ids = [i for i in sorted(self.data.get(namespace or "", {})) if i.startswith(prefix)]
        for start in range(0, len(ids), self.page_size):
            yield ids[start : start + self.page_size]

    def delete(self, ids: list[str], namespace: str | None = None) -> None:
        for i in ids:
            self.data.get(namespace or "", {}).pop(i, None)

    def describe_index_stats(self) -> dict[str, Any]:
        return {"namespaces": {ns: {"vector_count": len(v)} for ns, v in self.data.items()}}


def make_chunk(
    i: int = 0, *, doc: str = "d1", version: int = 1, access: str = "internal",
    page: int | None = None, section: str | None = None, text: str = "some text",
) -> Chunk:
    """Build a Chunk for tests."""
    return Chunk(
        chunk_id=f"{doc}:v{version}:{i}", document_id=doc, document_name=f"{doc}.txt", title="Doc",
        chunk_index=i, text=text, category="general", department="IT", access_level=access,  # type: ignore[arg-type]
        page_number=page, section=section, uploaded_at="2026-01-01T00:00:00+00:00", version=version,
    )


def make_docx_bytes(items: list[tuple[str, str]]) -> bytes:
    """Build a DOCX. Items are (kind, text) with kind in {'title', 'h1', 'p'}."""
    from docx import Document

    doc = Document()
    for kind, text in items:
        if kind == "title":
            doc.add_heading(text, 0)
        elif kind == "h1":
            doc.add_heading(text, 1)
        else:
            doc.add_paragraph(text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def make_pdf_bytes(pages: list[str]) -> bytes:
    """Build a minimal text PDF with one single-line text per page."""
    n = len(pages)
    page_ids = [4 + 2 * i for i in range(n)]
    objs: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{' '.join(f'{p} 0 R' for p in page_ids)}] /Count {n} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(pages):
        content_id = page_ids[i] + 1
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 3 0 R >> >> >>".encode()
        )
        esc = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({esc}) Tj ET".encode()
        objs.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for num, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{num} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


# ---- Phase 3: LLM fakes ---------------------------------------------------------------
from types import SimpleNamespace  # noqa: E402

from copilot.llm.client import LLMUsage  # noqa: E402


class FakeGroqSDK:
    """Mimics ``client.chat.completions.create`` with scripted outcomes (str reply or Exception)."""

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model="fake-model",
        )


class FakeLLM:
    """Stands in for GroqClient.structured(): scripted dicts (validated by the schema) or exceptions."""

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[list[dict[str, str]]] = []

    def structured(self, messages, schema, *, max_tokens=None):
        self.calls.append([dict(m) for m in messages])
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return schema.model_validate(outcome), LLMUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120, calls=1)


# ---- Phase 4: scripted LLM and fake search ----------------------------------------------
class ScriptedLLM:
    """LLM double for graph tests: scripted outputs per schema name, consumed in order.

    The LAST item of each script repeats, so ``PrivateGrade=[BAD]`` means "always BAD".
    Items may be dicts (validated against the requested schema) or exceptions to raise.
    Calling a schema with no script fails the test, so unexpected LLM calls are caught.
    """

    def __init__(self, **scripts: list[Any]) -> None:
        self.scripts = {name: list(items) for name, items in scripts.items()}
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def structured(self, messages, schema, *, max_tokens=None):
        name = schema.__name__
        self.calls.append((name, [dict(m) for m in messages]))
        queue = self.scripts.get(name)
        if not queue:
            raise AssertionError(f"Unexpected LLM call for schema {name}")
        outcome = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return schema.model_validate(outcome), LLMUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120, calls=1)

    def count(self, name: str) -> int:
        return sum(1 for n, _ in self.calls if n == name)

    def names(self) -> list[str]:
        return [n for n, _ in self.calls]

    def prompt(self, name: str, index: int = 0) -> str:
        """Concatenated message contents of the ``index``-th call to schema ``name``."""
        matching = [msgs for n, msgs in self.calls if n == name]
        return "\n".join(m["content"] for m in matching[index])


class FakeSearch:
    """Search double: records sanitised queries; can return results or raise."""

    def __init__(self, results=None, error: BaseException | None = None) -> None:
        self.results = list(results or [])
        self.error = error
        self.queries: list[str] = []
        self.call_count = 0

    def search(self, query: str):
        self.queries.append(query)
        self.call_count += 1
        if self.error:
            raise self.error
        return self.results
