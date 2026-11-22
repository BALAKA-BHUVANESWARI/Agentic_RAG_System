"""Evaluation runner: runs the baseline RAG and/or the agentic workflow (or named ablation variants of it) over the labelled dataset.

Every system sees the SAME questions, roles and corpus. Cases run sequentially (shared provider rate
limits) and every result is appended to ``cases.jsonl`` the moment it exists, so an interrupted run can be
resumed: finished cases are skipped, failed ones are retried. Per run directory
(``<EVAL_DIR>/results/<run_id>/``): ``config.json``, ``retrieval.jsonl``, ``cases.jsonl``, ``summary.json``,
``cases.csv``, ``metrics.csv``.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, TypeVar

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from copilot.config import Settings
from copilot.db.schema import init_db
from copilot.evaluation import adapters
from copilot.evaluation import split as split_registry
from copilot.evaluation import variants as variant_registry
from copilot.evaluation.aggregate import ablation_table, compare_modes, summarize_mode, summarize_retrieval
from copilot.evaluation.attacks import is_attack, summarize_attacks
from copilot.evaluation.paired import build_intervals
from copilot.evaluation.reporting import write_cases_csv, write_metrics_csv
from copilot.evaluation.results import CaseResult, RetrievalCaseResult
from copilot.evaluation.retrieval_metrics import rank_case
from copilot.evaluation.schemas import CATEGORIES, EvalCase, load_dataset, validate_dataset
from copilot.evaluation.scoring import score_case
from copilot.evaluation.store import EvalRunStore
from copilot.evaluation.system import file_sha256, ingestion_stats, settings_snapshot
from copilot.logging_config import get_logger, redact
from copilot.utils import utc_now_iso

log = get_logger(__name__)
M = TypeVar("M", bound=BaseModel)

RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
ProgressFn = Callable[[int, int, str], None]  # (units done, units total, message)


def new_run_id() -> str:
    """Sortable, unique, filesystem-safe id such as ``20261117T081500Z-a1b2c3``."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


class RunOptions(BaseModel):
    """What to run. ``resume=True`` re-uses the options saved with ``run_id`` (other selection fields are ignored)."""

    mode: Literal["agentic", "baseline", "both"] = "both"
    split: Literal["all", "dev", "test"] = Field(default="all", description="Frozen dev/test split (docs/EVAL_SPLIT.md); 'all' runs every case.")
    variants: list[str] | None = Field(default=None, description="Ablation ladder rungs to run instead of ``mode`` (see evaluation/variants.py).")
    limit: int | None = Field(default=None, ge=1, description="First N cases after the other filters.")
    per_category: int | None = Field(default=None, ge=1, description="First N cases of EACH category (good for cheap smoke runs).")
    categories: list[str] | None = None
    attacks_only: bool = Field(default=False, description="Only the adversarial block (notes start with 'ADVERSARIAL:'), e.g. for an attack-success run.")
    use_judge: bool = True
    run_id: str | None = None
    resume: bool = False
    case_delay_s: float | None = Field(default=None, ge=0, description="Overrides EVAL_CASE_DELAY_S.")

    @field_validator("categories")
    @classmethod
    def _known_categories(cls, v: list[str] | None) -> list[str] | None:
        bad = sorted(set(v or []) - set(CATEGORIES))
        if bad:
            raise ValueError(f"unknown categories {bad}; valid: {CATEGORIES}")
        return v or None

    @field_validator("variants")
    @classmethod
    def _known_variants(cls, v: list[str] | None) -> list[str] | None:
        return variant_registry.normalise(v) if v else None

    @model_validator(mode="after")
    def _mode_or_variants(self) -> "RunOptions":
        if self.variants and self.mode != "both":
            raise ValueError("use either `mode` or `variants`, not both")
        return self

    @property
    def modes(self) -> list[str]:
        """Systems to run, baseline first (ablation rungs in ladder order when ``variants`` is set)."""
        if self.variants:
            return list(self.variants)
        return ["baseline", "agentic"] if self.mode == "both" else [self.mode]

    @property
    def store_mode(self) -> str:
        """Value for the ``eval_runs.mode`` column, whose CHECK constraint allows only agentic / baseline / both.

        The exact rungs of an ablation run are saved in the run's metadata and ``config.json``.
        """
        modes = self.modes
        if "baseline" in modes:
            return "both" if len(modes) > 1 else "baseline"
        return "agentic"


def select_cases(cases: list[EvalCase], opts: RunOptions) -> list[EvalCase]:
    """Apply category / per-category / limit filters (in that order). Raises if nothing is left."""
    sel = [c for c in cases if (not opts.categories or c.category in opts.categories) and (not opts.attacks_only or is_attack(c))]
    if opts.per_category:
        seen: dict[str, int] = {}
        picked = []
        for c in sel:
            if seen.get(c.category, 0) < opts.per_category:
                picked.append(c)
                seen[c.category] = seen.get(c.category, 0) + 1
        sel = picked
    if opts.limit:
        sel = sel[: opts.limit]
    if not sel:
        raise ValueError("No evaluation cases match the selected filters.")
    return sel


def _read_jsonl(path: Path, model: type[M]) -> Iterator[M]:
    if not path.exists():
        return
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            yield model.model_validate_json(line)
        except ValidationError:  # e.g. a line cut short by a crash; the case simply runs again
            log.warning("Skipping unreadable line %d in %s", n, path.name)


def _append_jsonl(path: Path, record: BaseModel) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")
        f.flush()
        os.fsync(f.fileno())


_REPLACE_ATTEMPTS = 20
_REPLACE_WAIT_SECONDS = 0.1


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    # Atomic: readers never see half a file. On Windows os.replace raises PermissionError while
    # another handle (the dashboard polling the file, an antivirus scan) is open on it, so retry
    # briefly instead of failing a finished run at its last step.
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(_REPLACE_WAIT_SECONDS)


def headline(summary: dict[str, Any]) -> dict[str, Any]:
    """Small metrics dict stored in the SQLite ``eval_runs.metrics`` column."""
    def v(m: Any) -> Any:
        return m.get("value") if isinstance(m, dict) else m

    out: dict[str, Any] = {}
    for mode, ms in (summary.get("modes") or {}).items():
        a, r, sy = ms.get("answer", {}), ms.get("routing", {}), ms.get("system", {})
        out[mode] = {
            "overall_correct_rate": v(a.get("overall_correct_rate")), "answer_correct_rate": v(a.get("answer_correct_rate")),
            "correct_abstention_rate": v(a.get("correct_abstention_rate")), "latency_avg_ms": sy.get("latency_avg_ms"),
            "latency_p95_ms": sy.get("latency_p95_ms"), "estimated_cost_usd": sy.get("estimated_cost_usd"),
            "error_rate": v(sy.get("error_rate")), "n_cases": sy.get("n_cases"),
        }
        if r:
            out[mode]["unnecessary_external_search_rate"] = v(r.get("unnecessary_external_search_rate"))
            out[mode]["private_first_routing_accuracy"] = v(r.get("private_first_routing_accuracy"))
    return out


class EvaluationRunner:
    """Runs and scores an evaluation. Dependencies are injected so tests can use stubs.

    ``agent`` needs ``answer(question, role, *, user_id, session_id, external_search_allowed)``; ``baseline`` needs
    ``answer(question, role)``; ``retriever`` needs ``retrieve(query, role, top_k=, min_score=)``; ``judge`` is an
    :class:`~copilot.evaluation.judge.AnswerJudge` or None.
    """

    def __init__(
        self, settings: Settings, *, agent: Any, baseline: Any, retriever: Any, judge: Any | None = None,
        store: EvalRunStore | None = None, validate: bool = True, sleep: Callable[[float], None] = time.sleep,
        agent_factory: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self._s = settings
        self._agent, self._baseline, self._retriever, self._judge = agent, baseline, retriever, judge
        # ``agent_factory(overrides)`` builds an agent with those Settings fields changed (ablation variants only).
        self._agent_factory = agent_factory
        self._variant_agents: dict[str, Any] = {}
        self._store = store or EvalRunStore(settings.sqlite_path)
        self._validate = validate
        self._sleep = sleep

    # ------------------------------------------------------------------ paths and saved state
    @property
    def results_root(self) -> Path:
        """Directory holding one sub-directory per run."""
        return self._s.eval_dir / "results"

    def run_dir(self, run_id: str) -> Path:
        """Run directory for a validated id (rejects anything that could escape the results folder)."""
        if not RUN_ID_RE.match(run_id or ""):
            raise ValueError("Invalid run id (use letters, digits, '_' and '-', max 64 characters).")
        root = self.results_root.resolve()
        d = (root / run_id).resolve()
        if d.parent != root:
            raise ValueError("Invalid run id.")
        return d

    def read_config(self, run_id: str) -> dict[str, Any]:
        """The ``config.json`` saved when the run started."""
        path = self.run_dir(run_id) / "config.json"
        if not path.exists():
            raise ValueError(f"Run '{run_id}' has no saved configuration, so it cannot be resumed.")
        return json.loads(path.read_text(encoding="utf-8"))

    def read_summary(self, run_id: str) -> dict[str, Any] | None:
        """The finished (or partial) ``summary.json``, or None."""
        try:
            path = self.run_dir(run_id) / "summary.json"
        except ValueError:
            return None
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def read_progress(self, run_id: str) -> dict[str, int]:
        """Units (case x mode) finished without error vs planned, from the files on disk."""
        try:
            cfg = self.read_config(run_id)
        except ValueError:
            return {"done": 0, "total": 0}
        ok = sum(1 for r in self._load_results(self.run_dir(run_id)).values() if r.error is None)
        return {"done": ok, "total": int(cfg.get("total_units", 0))}

    # ------------------------------------------------------------------ planning
    def prepare(self, opts: RunOptions) -> tuple[RunOptions, list[EvalCase]]:
        """Resolve resume options, load and validate the dataset, select the cases. Raises ``ValueError`` on bad input."""
        if opts.resume:
            if not opts.run_id:
                raise ValueError("Resuming needs the run id.")
            saved = RunOptions.model_validate(self.read_config(opts.run_id)["options"])
            opts = saved.model_copy(update={"run_id": opts.run_id, "resume": True, "case_delay_s": opts.case_delay_s})
        path = self._s.eval_dataset_path
        if not path.exists():
            raise ValueError(f"Evaluation dataset not found: {path}")
        cases = load_dataset(path)
        if self._validate:
            problems = validate_dataset(cases, self._s.kb_documents_dir)
            if problems:
                raise ValueError("Evaluation dataset has problems: " + "; ".join(problems[:5]))
        if opts.resume and self.read_config(opts.run_id or "").get("dataset_sha256") != file_sha256(path):
            raise ValueError("The dataset changed since this run started; start a new run instead of resuming.")
        if self._agent_factory is None and any(m not in ("baseline", "agentic") for m in opts.modes):
            raise ValueError("Ablation variants need a runner built with an agent factory (use scripts/run_eval.py --variants).")
        if opts.split != "all":
            cases = self._cases_in_split(cases, opts)
        return opts, select_cases(cases, opts)

    def _cases_in_split(self, cases: list[EvalCase], opts: RunOptions) -> list[EvalCase]:
        """The cases of the chosen split; refuses a split file that is inconsistent with the dataset or changed since a resumed run began."""
        path = self._s.eval_split_path
        sp = split_registry.load_split(path)  # ValueError: missing or malformed
        problems = sp.problems({c.id: c.category for c in cases})
        if problems:
            raise ValueError("Evaluation split has problems: " + "; ".join(problems[:3]))
        if opts.resume and self.read_config(opts.run_id or "").get("split_sha256") != file_sha256(path):
            raise ValueError("The split file changed since this run started; start a new run instead of resuming.")
        ids = sp.ids(opts.split)
        return [c for c in cases if c.id in ids]

    # ------------------------------------------------------------------ the run
    def run(self, opts: RunOptions, *, progress: ProgressFn | None = None, should_cancel: Callable[[], bool] | None = None) -> dict[str, Any]:
        """Execute (or resume) a run and return its summary. Per-case failures never abort the run."""
        opts, cases = self.prepare(opts)
        s = self._s
        modes = opts.modes
        run_id = opts.run_id or new_run_id()
        run_dir = self.run_dir(run_id)
        if not opts.resume and run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError(f"Run '{run_id}' already exists; resume it or choose another id.")
        run_dir.mkdir(parents=True, exist_ok=True)
        init_db(s.sqlite_path)

        if opts.resume:
            cfg = self.read_config(run_id)
        else:
            cfg = {
                "run_id": run_id, "started_at": utc_now_iso(), "options": opts.model_copy(update={"run_id": run_id, "resume": False}).model_dump(mode="json"),
                "dataset_path": str(s.eval_dataset_path), "dataset_sha256": file_sha256(s.eval_dataset_path),
                "case_ids": [c.id for c in cases], "total_units": len(cases) * len(modes), "settings": settings_snapshot(s),
            }
            if opts.variants:
                cfg["variants"] = {m: variant_registry.overrides_for(m) for m in modes}
            cfg["split_sha256"] = file_sha256(s.eval_split_path) if opts.split != "all" else None
            _write_json(run_dir / "config.json", cfg)
        self._store.start_run(
            run_id, opts.store_mode, str(s.eval_dataset_path), len(cases),
            {"use_judge": opts.use_judge and self._judge is not None, "dataset_sha256": cfg["dataset_sha256"], "resumed": opts.resume,
             "categories": opts.categories, "attacks_only": opts.attacks_only, "limit": opts.limit, "per_category": opts.per_category, "variants": opts.variants},
        )

        status, error = "completed", None
        try:
            retrieval = self._retrieval_pass(cases, run_dir)
            outcome = self._case_pass(cases, modes, run_id, run_dir, opts, progress, should_cancel)
            if outcome == "cancelled":
                status = "cancelled"
            elif outcome == "stopped":
                status = "stopped"
                error = (
                    f"Stopped after {self._s.eval_max_consecutive_errors} consecutive infrastructure errors "
                    f"(likely a provider quota or outage). Resume later: --resume {run_id}"
                )
        except Exception as exc:  # noqa: BLE001 - a bug or disk error: keep what exists and say so
            log.exception("Evaluation run failed")
            status, error = "failed", redact(f"{type(exc).__name__}: {exc}")[:300]
            retrieval = {r.case_id: r for r in _read_jsonl(run_dir / "retrieval.jsonl", RetrievalCaseResult)}
        return self._finalize(run_id, run_dir, opts, cases, modes, retrieval, cfg, status, error)

    # ------------------------------------------------------------------ passes
    def _retrieval_pass(self, cases: list[EvalCase], run_dir: Path) -> dict[str, RetrievalCaseResult]:
        """Rank a pool of chunks per question (no LLM) and compute retrieval metrics.

        Uses ``min_score=0`` so ranking quality is measured over the whole pool; the first K items are what a
        top-K retriever returns for the ORIGINAL question (query rewriting is not part of this measurement).
        The measurement itself is :func:`copilot.evaluation.retrieval_metrics.rank_case`.
        """
        path = run_dir / "retrieval.jsonl"
        done = {r.case_id: r for r in _read_jsonl(path, RetrievalCaseResult)}
        for case in cases:
            if not case.expected_documents or case.id in done:
                continue
            try:
                rcr = rank_case(self._retriever, case, self._s.eval_retrieval_pool, self._s.eval_ks)
            except Exception as exc:  # noqa: BLE001 - leave it out; a resumed run retries it
                log.warning("Retrieval failed for %s: %s", case.id, type(exc).__name__)
                continue
            _append_jsonl(path, rcr)
            done[case.id] = rcr
        wanted = {c.id for c in cases}
        return {cid: r for cid, r in done.items() if cid in wanted}

    def _case_pass(self, cases, modes, run_id, run_dir, opts, progress, should_cancel) -> str:
        """Run every (case, mode) not yet finished.

        Returns ``"completed"``, ``"cancelled"`` or ``"stopped"`` (too many consecutive infrastructure errors).
        """
        path = run_dir / "cases.jsonl"
        wanted = {(c.id, m) for c in cases for m in modes}
        finished = {k for k, r in self._load_results(run_dir).items() if r.error is None and k in wanted}
        total, done, ran = len(wanted), len(finished), 0
        delay = self._s.eval_case_delay_s if opts.case_delay_s is None else opts.case_delay_s
        self._emit(progress, done, total, "resuming" if done else "starting")
        streak, limit = 0, self._s.eval_max_consecutive_errors
        for case in cases:
            for mode in modes:
                if (case.id, mode) in finished:
                    continue
                if should_cancel and should_cancel():
                    return "cancelled"
                if streak >= limit:
                    return "stopped"
                if ran and delay:
                    self._sleep(delay)
                result = self._run_one(case, mode, run_id, opts.use_judge)
                _append_jsonl(path, result)
                ran, done = ran + 1, done + 1
                streak = streak + 1 if result.error else 0
                self._emit(progress, done, total, f"{case.id} [{mode}] {result.scores.get('outcome')}")
        return "stopped" if streak >= limit else "completed"

    @staticmethod
    def _emit(progress: ProgressFn | None, done: int, total: int, message: str) -> None:
        if progress:
            try:
                progress(done, total, message)
            except Exception:  # noqa: BLE001 - a broken progress callback must not stop the evaluation
                log.warning("Progress callback failed")

    def _agent_for(self, mode: str) -> Any:
        """``agentic`` is the injected full agent; an ablation variant is built once, with its settings overrides."""
        if mode == "agentic":
            return self._agent
        if mode not in self._variant_agents:
            if self._agent_factory is None:
                raise ValueError(f"variant {mode!r} needs an agent factory")
            self._variant_agents[mode] = self._agent_factory(variant_registry.overrides_for(mode))
        return self._variant_agents[mode]

    def _run_one(self, case: EvalCase, mode: str, run_id: str, use_judge: bool) -> CaseResult:
        """Ask one system one question, optionally judge, then score."""
        started = time.perf_counter()
        ans, evidence = None, []
        try:
            if mode == "baseline":
                result, ans, evidence = adapters.from_baseline(case, self._baseline.answer(case.question, case.role))
            else:
                resp = self._agent_for(mode).answer(
                    case.question, case.role, user_id="eval", session_id=run_id, external_search_allowed=case.external_search_allowed,
                )
                result, ans, evidence = adapters.from_agentic(case, resp, mode)
        except Exception as exc:  # noqa: BLE001 - recorded as an error result
            log.warning("Case %s [%s] raised %s", case.id, mode, type(exc).__name__)
            result = adapters.error_result(case, mode, exc, (time.perf_counter() - started) * 1000)

        judge_usage = None
        if use_judge and self._judge is not None and ans is not None and result.error is None and not result.abstained and evidence and ans.answer.strip():
            result.judge, judge_usage = self._judge.judge(case.question, ans.answer, evidence)
        s = self._s
        result.scores = score_case(case, result, answer_threshold=s.eval_answer_threshold, faithfulness_threshold=s.eval_faithfulness_threshold)
        if judge_usage is not None and judge_usage.calls:
            result.scores["judge_tokens"], result.scores["judge_calls"] = judge_usage.total_tokens, judge_usage.calls
        return result

    # ------------------------------------------------------------------ results and summary
    @staticmethod
    def _load_results(run_dir: Path) -> dict[tuple[str, str], CaseResult]:
        """Latest result per (case, mode); a retried case replaces its earlier failed attempt."""
        return {(r.case_id, r.mode): r for r in _read_jsonl(run_dir / "cases.jsonl", CaseResult)}

    def _finalize(self, run_id, run_dir, opts, cases, modes, retrieval, cfg, status, error) -> dict[str, Any]:
        s = self._s
        wanted = {(c.id, m) for c in cases for m in modes}
        results = {k: r for k, r in self._load_results(run_dir).items() if k in wanted}
        per_mode = {m: [r for (cid, mm), r in results.items() if mm == m] for m in modes}
        notes = [
            f"Retrieval metrics rank a pool of {s.eval_retrieval_pool} chunks for the original question (minimum score ignored); chunk recall is relative to that pool.",
            "All results are from a single run on a small labelled set; differences of a few points can be noise.",
        ]
        if len(modes) >= 2 and all(per_mode[m] for m in modes):
            paired = set.intersection(*({r.case_id for r in per_mode[m]} for m in modes))
            if len(paired) < max(len(per_mode[m]) for m in modes):
                who = " and ".join(modes) if len(modes) == 2 else ", ".join(modes)
                notes.append(f"The run is incomplete: {who} are compared only on the {len(paired)} question(s) {'both' if len(modes) == 2 else 'all of them'} have answered.")
            per_mode = {m: [r for r in rs if r.case_id in paired] for m, rs in per_mode.items()}
        modes_summary = {m: summarize_mode(cases, rs, s) for m, rs in per_mode.items() if rs}
        if opts.split == "test":
            notes.append("This run used the frozen TEST split: report it, but do not change prompts, thresholds or routing because of an individual failure (see docs/EVAL_SPLIT.md).")
        elif opts.split == "dev":
            notes.append("This run used the DEV split: fine for tuning, but quote the test split for a headline number.")
        errors = sum(1 for r in results.values() if r.error)
        if errors:
            notes.append(f"{errors} request(s) failed (service errors) and count as incorrect. Resume the run to retry them.")
        expected_with_docs = sum(1 for c in cases if c.expected_documents)
        if len(retrieval) < expected_with_docs:
            notes.append(f"Retrieval was measured for {len(retrieval)} of {expected_with_docs} cases (the rest failed; resume to retry).")
        judged = [r for r in results.values() if r.scores.get("judge_calls")]
        if opts.use_judge and self._judge is not None:
            notes.append("Faithfulness/relevance come from an LLM judge (same model family as the generator); treat them as a noisy signal.")
        summary: dict[str, Any] = {
            "run_id": run_id, "status": status, "error": error, "options": cfg["options"],
            "dataset": {"path": cfg["dataset_path"], "sha256": cfg["dataset_sha256"], "num_cases": len(cases), "split": opts.split,
                        "category_counts": {c: sum(1 for x in cases if x.category == c) for c in CATEGORIES if any(x.category == c for x in cases)}},
            "started_at": cfg["started_at"], "finished_at": utc_now_iso(), "settings": cfg["settings"],
            "progress": {"units_total": len(wanted), "units_recorded": len(results), "units_ok": sum(1 for r in results.values() if r.error is None)},
            "retrieval": summarize_retrieval(list(retrieval.values()), s.eval_ks) if retrieval else None,
            "modes": modes_summary,
            "comparison": compare_modes(modes_summary["baseline"], modes_summary["agentic"]) if {"baseline", "agentic"} <= set(modes_summary) else None,
            "variants": cfg.get("variants"),
            "intervals": build_intervals(cases, per_mode, modes, s, is_ablation=bool(opts.variants)),
            "attacks": summarize_attacks(cases, per_mode, s.eval_confidence),
            "ablation": ablation_table(modes_summary, modes, {m: variant_registry.get_variant(m).label for m in modes}) if opts.variants and len(modes_summary) > 1 else None,
            "ingestion": ingestion_stats(s.sqlite_path),
            "judge": {"enabled": bool(opts.use_judge and self._judge is not None), "judged_answers": len(judged),
                      "tokens": sum(r.scores.get("judge_tokens", 0) for r in judged), "calls": sum(r.scores.get("judge_calls", 0) for r in judged)},
            "failed_cases": self._failed_cases(cases, results),
            "notes": notes,
        }
        _write_json(run_dir / "summary.json", summary)
        write_cases_csv(run_dir / "cases.csv", cases, list(results.values()))
        write_metrics_csv(run_dir / "metrics.csv", summary)
        self._store.finish_run(run_id, status=status, metrics=headline(summary), results_path=str(run_dir),
                               extra={"error": error, "units_ok": summary["progress"]["units_ok"], "units_total": len(wanted)})
        log.info("Evaluation run finished", extra={"run_id": run_id, "status": status})
        return summary

    @staticmethod
    def _failed_cases(cases: list[EvalCase], results: dict[tuple[str, str], CaseResult]) -> list[dict[str, Any]]:
        by_id = {c.id: c for c in cases}
        out = []
        for (cid, mode), r in sorted(results.items()):
            if r.scores.get("correct") and r.error is None:
                continue
            c = by_id[cid]
            out.append({
                "case_id": cid, "category": c.category, "mode": mode, "question": c.question, "expected_route": c.expected_route,
                "outcome": r.scores.get("outcome", "error"), "abstention_reason": r.abstention_reason, "error": r.error,
                "points_met": r.scores.get("points_met"), "points_total": r.scores.get("points_total"),
                "cited_documents": r.cited_documents, "path": r.path, "answer": r.answer[:600], "request_id": r.request_id,
            })
        return out
