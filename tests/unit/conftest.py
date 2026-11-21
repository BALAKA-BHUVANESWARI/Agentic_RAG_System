"""Fixtures shared by the evaluation unit tests."""
from __future__ import annotations

import pytest

from copilot.evaluation.runner import EvaluationRunner
from eval_helpers import StubAgent, StubBaseline, write_dataset


@pytest.fixture
def make_runner(settings, retriever, kb, tmp_path):
    """Factory: runner over the real (fake-embedded) KB with stub agent/baseline. Returns (runner, agent, baseline, sleeps)."""

    def _make(agent=None, baseline=None, judge=None, validate=False, agent_factory=None, **overrides):
        s = settings.model_copy(update={"eval_dir": tmp_path / "eval", "eval_dataset_path": write_dataset(tmp_path / "ds.json"), **overrides})
        agent, baseline, sleeps = agent or StubAgent(), baseline or StubBaseline(), []
        runner = EvaluationRunner(
            s, agent=agent, baseline=baseline, retriever=retriever, judge=judge, validate=validate, sleep=sleeps.append, agent_factory=agent_factory,
        )
        return runner, agent, baseline, sleeps

    return _make
