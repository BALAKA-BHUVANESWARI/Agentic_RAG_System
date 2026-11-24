"""Centralized configuration for the IT Support Copilot.

This is the ONLY module that reads environment variables or ``.env``. Every other
module obtains settings through :func:`get_settings`. Nothing else in the code base
may hardcode API keys, model names, thresholds, top-k, chunk sizes or paths.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

AccessLevel = Literal["public", "internal", "restricted"]


class Settings(BaseSettings):
    """Application settings loaded from environment variables / ``.env``.

    Field names map to upper-case env vars (``groq_api_key`` -> ``GROQ_API_KEY``).
    List/dict values must be provided as JSON when set through the environment.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- General -------------------------------------------------------
    app_env: Literal["local", "test"] = "local"
    log_level: str = "INFO"
    log_json: bool = True

    # ---- Secrets (never logged, never sent to the frontend) -------------
    groq_api_key: SecretStr | None = None
    pinecone_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None

    # ---- Groq LLM ------------------------------------------------------
    groq_model: str = "openai/gpt-oss-120b"
    groq_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    # Nucleus sampling. None = provider default (1.0). Only sampled tokens are affected, so with
    # temperature 0 it changes little; it is a determinism guard, not a hallucination fix.
    groq_top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    groq_timeout_s: float = Field(default=30.0, gt=0)
    groq_max_retries: int = Field(default=2, ge=0, le=5)
    # Reasoning models (e.g. openai/gpt-oss-*) accept low|medium|high; leave empty for other models.
    groq_reasoning_effort: Literal["low", "medium", "high"] | None = "low"
    llm_max_tokens: int = Field(default=2048, ge=64)
    evidence_max_chars: int = Field(default=3000, ge=200)

    # ---- Embeddings ----------------------------------------------------
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = Field(default=384, gt=0)

    # ---- Pinecone ------------------------------------------------------
    pinecone_index_name: str = "it-support-copilot"
    pinecone_namespace: str = "kb"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_metric: Literal["cosine"] = "cosine"

    # ---- Tavily / external search --------------------------------------
    external_search_enabled: bool = True
    tavily_max_results: int = Field(default=5, ge=1, le=10)
    tavily_search_depth: Literal["basic", "advanced"] = "basic"
    tavily_timeout_s: float = Field(default=15.0, gt=0)
    user_generated_url_markers: list[str] = Field(
        default_factory=lambda: ["/answers/", "/discussions/", "/forums/", "/forum/", "/community/", "/t5/", "/questions/"],
        description="URL path fragments that mark community content; such pages are never treated as official.",
    )
    preferred_external_domains: list[str] = Field(
        default_factory=lambda: [
            "learn.microsoft.com",
            "support.microsoft.com",
            "status.cloud.microsoft",
            "techcommunity.microsoft.com",
        ]
    )

    external_content_max_chars: int = Field(default=1500, ge=200)
    max_external_query_chars: int = Field(default=200, ge=20)
    # Internal names stripped from any query sent to public search (override via JSON in env).
    private_terms: list[str] = Field(default_factory=lambda: ["Acme Demo Corp", "Acme", "SecureLink"])

    # ---- Retrieval + grading thresholds --------------------------------
    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    retrieval_min_score: float = Field(default=0.25, ge=0.0, le=1.0)
    # Optional second-stage reranker (local cross-encoder). Off by default so existing evaluation numbers stay valid.
    rerank_enabled: bool = False
    rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = Field(default=20, ge=1, le=50, description="Chunks fetched from Pinecone before reranking or per-document capping down to top_k.")
    retrieval_max_chunks_per_doc: int | None = Field(
        default=None, ge=1, le=50,
        description="Optional soft cap on chunks from one document in the final top-k (None = off). Frees slots for other documents; unused slots are refilled.",
    )
    private_evidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    external_evidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    external_max_age_days: int = Field(default=365, gt=0)

    # ---- Loop bounds ----------------------------------------------------
    max_rewrite_retries: int = Field(default=2, ge=0, le=5)
    max_validation_revisions: int = Field(default=1, ge=0, le=3)
    graph_recursion_limit: int = Field(default=40, ge=10, le=200)
    # ABLATION switches for evaluation only: they measure what each agent step buys. Every default keeps the full workflow.
    agent_use_grader: bool = Field(default=True, description="False skips the LLM evidence grader: any retrieved chunk counts as sufficient (nothing retrieved still abstains).")
    agent_use_validator: bool = Field(default=True, description="False skips the final LLM grounding check; citation marker checks in the answer generator still run.")
    # EXPERIMENTS A and B (docs/POST_DAY3_EXPERIMENTS.md): default OFF so every recorded run keeps its meaning.
    retrieval_query_expansion: int = Field(
        default=0, ge=0, le=2,
        description="EXPERIMENT C: 0 = off. 1 or 2 = before the FIRST private retrieval only, one extra LLM call writes that many "
                    "short paraphrases of the question; retrieval runs for the original and each paraphrase and the results are merged "
                    "by best score. The grader and the answer generator still see only the original question.",
    )
    partial_private_evidence: bool = Field(
        default=False,
        description="True: when the classifier intent is 'both', the private grader found supporting chunks with confidence at or above "
                    "the private threshold but marked the evidence insufficient only because the question also needs current public "
                    "information, those supporting chunks still reach the answer generator (never a retrieval path of their own).",
    )
    external_label_check_in_code: bool = Field(
        default=False,
        description="True: for an answer that cites ONLY external sources and starts with the required external-information line, "
                    "the separation of public vendor information from company guidance is decided in code and the LLM validator's "
                    "external_clearly_separated field is ignored. Answers citing both private and external sources still use the LLM judgement.",
    )

    # ---- Embeddings batching / external-call resilience ----------------
    embedding_batch_size: int = Field(default=32, ge=1)
    pinecone_upsert_batch_size: int = Field(default=100, ge=1, le=1000)
    service_max_retries: int = Field(default=2, ge=0, le=5)
    service_retry_base_delay_s: float = Field(default=0.5, ge=0.0)
    service_retry_max_delay_s: float = Field(default=30.0, ge=0.0)

    # ---- Ingestion ------------------------------------------------------
    chunk_size: int = Field(default=800, ge=100)
    chunk_overlap: int = Field(default=120, ge=0)
    max_upload_mb: int = Field(default=10, ge=1)
    allowed_extensions: list[str] = Field(default_factory=lambda: [".pdf", ".docx", ".txt"])

    # ---- API -------------------------------------------------------------
    max_question_chars: int = Field(default=2000, ge=50, le=10000)
    max_comment_chars: int = Field(default=1000, ge=0)
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_base_url: str = "http://127.0.0.1:8000"  # where the Streamlit UI finds the API
    ui_request_timeout_s: float = Field(default=180.0, gt=0)
    ui_eval_refresh_s: float = Field(default=5.0, ge=1.0, description="Seconds between automatic progress refreshes of a running evaluation in the dashboard.")

    # ---- Evaluation -----------------------------------------------------
    eval_dataset_path: Path = Path("data/eval/dataset.json")
    eval_split_path: Path = Path("data/eval/split.json")  # frozen dev/test assignment, see docs/EVAL_SPLIT.md
    eval_retrieval_baseline_path: Path = Path("data/eval/retrieval_baseline.json")  # committed baseline of the retrieval regression check
    eval_regression_tolerance: float = Field(default=0.02, ge=0.0, le=0.5, description="Largest drop in a mean retrieval metric the regression check accepts (absolute).")
    eval_ks: list[int] = Field(default_factory=lambda: [1, 3, 5])
    eval_retrieval_pool: int = Field(default=20, ge=5, le=50, description="Chunks retrieved to estimate how many relevant chunks exist.")
    eval_answer_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Share of expected answer points an answer must contain to count as correct.")
    eval_faithfulness_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    eval_case_delay_s: float = Field(default=0.0, ge=0.0, description="Pause between cases (helps with provider rate limits).")
    eval_bootstrap_resamples: int = Field(default=2000, ge=200, le=20000, description="Bootstrap resamples for the paired-difference intervals.")
    eval_bootstrap_seed: int = Field(default=7, description="Seed for the bootstrap, so a run always reports the same intervals.")
    eval_confidence: float = Field(default=0.95, ge=0.8, le=0.99, description="Confidence level of the reported intervals.")
    eval_max_consecutive_errors: int = Field(
        default=3, ge=1,
        description="Stop an evaluation run after this many infrastructure errors in a row (a quota or outage would only waste time); resume later.",
    )
    # Approximate list prices in USD (edit if your plan differs); used only for the cost estimate.
    price_llm_input_per_mtok: float = Field(default=0.15, ge=0.0)
    price_llm_output_per_mtok: float = Field(default=0.60, ge=0.0)
    price_tavily_per_search: float = Field(default=0.008, ge=0.0)

    # ---- Paths ----------------------------------------------------------
    sqlite_path: Path = Path("data/sqlite/app.db")
    kb_documents_dir: Path = Path("data/kb_documents")
    eval_dir: Path = Path("data/eval")
    uploads_dir: Path = Path("data/uploads")  # raw source files kept for re-indexing

    # ---- Local demo authorization (NOT production identity management) --
    default_role: str = "employee"
    role_access_levels: dict[str, list[AccessLevel]] = Field(
        default_factory=lambda: {
            "employee": ["public", "internal"],
            "admin": ["public", "internal", "restricted"],
        }
    )

    # ---- Validators -----------------------------------------------------
    @field_validator("sqlite_path", "kb_documents_dir", "eval_dir", "uploads_dir", "eval_dataset_path", "eval_split_path", "eval_retrieval_baseline_path")
    @classmethod
    def _resolve_relative_to_project(cls, value: Path) -> Path:
        """Anchor relative paths at the project root so cwd never matters."""
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @field_validator("allowed_extensions")
    @classmethod
    def _normalize_extensions(cls, value: list[str]) -> list[str]:
        return [e.lower() if e.startswith(".") else f".{e.lower()}" for e in value]

    @field_validator("groq_top_p", "retrieval_max_chunks_per_doc", mode="before")
    @classmethod
    def _empty_optional_number_is_none(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("groq_reasoning_effort", mode="before")
    @classmethod
    def _empty_reasoning_effort_is_none(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"Invalid LOG_LEVEL: {value}")
        return level

    @model_validator(mode="after")
    def _cross_field_checks(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        missing = {"employee", "admin"} - set(self.role_access_levels)
        if missing:
            raise ValueError(f"role_access_levels must define roles: {sorted(missing)}")
        if self.default_role not in self.role_access_levels:
            raise ValueError(f"DEFAULT_ROLE '{self.default_role}' is not a defined role")
        return self

    # ---- Helpers --------------------------------------------------------
    @property
    def max_upload_bytes(self) -> int:
        """Maximum accepted upload size in bytes."""
        return self.max_upload_mb * 1024 * 1024

    def access_levels_for(self, role: str) -> list[AccessLevel]:
        """Return the access levels a role may read. Unknown roles get none."""
        return list(self.role_access_levels.get(role, []))

    def secret_values(self) -> list[str]:
        """Plain-text secrets, used ONLY by the log redactor."""
        secrets = (self.groq_api_key, self.pinecone_api_key, self.tavily_api_key)
        return [s.get_secret_value() for s in secrets if s and s.get_secret_value()]

    def require_secret(self, name: Literal["groq", "pinecone", "tavily"]) -> str:
        """Return a configured API key or raise a clear, key-free error."""
        secret: SecretStr | None = getattr(self, f"{name}_api_key")
        if secret is None or not secret.get_secret_value():
            raise RuntimeError(
                f"{name.upper()}_API_KEY is not set. Add it to your .env file (see .env.example)."
            )
        return secret.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached :class:`Settings` instance."""
    return Settings()
