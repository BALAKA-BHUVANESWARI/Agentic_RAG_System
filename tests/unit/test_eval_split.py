"""The frozen dev/test split: assignment rules, integrity checks, the real split file, the runner's --split and the CLI. No network."""
from __future__ import annotations

import json
import random
import subprocess
import sys

import pytest
from pydantic import ValidationError

from copilot.config import PROJECT_ROOT
from copilot.evaluation import split as sp
from copilot.evaluation.runner import RunOptions
from eval_helpers import UNANSWERABLE_Q

SALT = "unit-test-salt"
# The test split as frozen on 2026-11-21. A case may only leave it through BURNED, with a reason (docs/EVAL_SPLIT.md).
FROZEN_TEST_IDS = {
    "A02", "C04", "C06", "D01", "D06", "D09", "E03", "E05", "I01", "I05", "M02", "M03", "O04", "P07", "P08", "R04",
    "R06", "R07", "U01", "U03", "U09", "V02", "X04", "X05", "X06", "X11", "X15", "X16",
}
BURNED: dict[str, str] = {}   # case id -> why it moved to dev (also record it in docs/FINDINGS.md)


def items(spec: dict[str, int]) -> list[tuple[str, str]]:
    """``{"direct_kb": 10}`` -> ids D01..D10 with that category (X-prefixed ids form the adversarial stratum)."""
    out = []
    for category, n in spec.items():
        prefix = {"direct_kb": "D", "unanswerable": "U", "adv": "X"}.get(category, category[:1].upper())
        out += [(f"{prefix}{i:02d}", "unanswerable" if category == "adv" else category) for i in range(1, n + 1)]
    return out


# ------------------------------------------------------------------------------ assignment rules
@pytest.mark.parametrize("size,expected", [(0, 0), (1, 0), (2, 0), (3, 1), (5, 2), (6, 2), (10, 3), (19, 6), (31, 9)])
def test_quota(size, expected):
    assert sp.quota(size) == expected


def test_adversarial_cases_are_their_own_stratum():
    assert sp.stratum_of("X07", "direct_kb") == "direct_kb/adversarial" and sp.stratum_of("D07", "direct_kb") == "direct_kb"


def test_plan_is_stratified_deterministic_and_independent_of_input_order():
    cases = items({"direct_kb": 10, "unanswerable": 6, "adv": 4, "conflicting": 2})
    a = sp.plan(cases, salt=SALT)
    shuffled = cases[:]
    random.Random(1).shuffle(shuffled)
    assert sp.plan(shuffled, salt=SALT) == a and sp.plan(cases, salt=SALT) == a
    assert sorted(a) == sorted(i for i, _ in cases)
    by_stratum: dict[str, list[str]] = {}
    for cid, cat in cases:
        by_stratum.setdefault(sp.stratum_of(cid, cat), []).append(a[cid])
    assert by_stratum["direct_kb"].count("test") == 3 and by_stratum["unanswerable"].count("test") == 2
    assert by_stratum["unanswerable/adversarial"].count("test") == 1                 # 4 cases -> 1
    assert by_stratum["conflicting"].count("test") == 0                              # under 3 cases stays in dev
    assert sp.plan(cases, salt="another salt") != a                                  # the salt decides, not the id order


def test_new_cases_never_move_old_assignments():
    first = sp.plan(items({"direct_kb": 10}), salt=SALT)
    grown = sp.plan(items({"direct_kb": 25}), first, salt=SALT)
    assert {i: grown[i] for i in first} == first                                     # frozen
    assert sum(1 for s in grown.values() if s == "test") == sp.quota(25)             # the stratum is topped up to its quota
    again = sp.plan(items({"direct_kb": 25}), grown, salt=SALT)
    assert again == grown                                                            # nothing left to decide


def test_a_frozen_assignment_with_extra_or_missing_ids_is_handled():
    frozen = {"D01": "test", "GONE": "test"}                                         # GONE is not in the dataset any more
    out = sp.plan(items({"direct_kb": 6}), frozen, salt=SALT)
    assert out["D01"] == "test" and "GONE" not in out and sum(1 for s in out.values() if s == "test") == 2


# ------------------------------------------------------------------------------ document, loading and integrity
def doc_for(assignment, **kw):
    return sp.build_document(assignment, salt=SALT, fraction=0.3, created="2026-01-01", **kw)


def write(tmp_path, doc, name="split.json"):
    path = tmp_path / name
    path.write_text(sp.dumps(doc), encoding="utf-8")
    return path


def test_document_round_trip_and_fingerprint(tmp_path):
    cases = items({"direct_kb": 10, "unanswerable": 6})
    assignment = sp.plan(cases, salt=SALT)
    doc = doc_for(assignment)
    assert doc["test"] == sorted(doc["test"]) and doc["dev"] == sorted(doc["dev"]) and doc["test_sha256"] == sp.fingerprint(doc["test"])
    loaded = sp.load_split(write(tmp_path, doc))
    assert loaded.assignment() == assignment and loaded.meta["salt"] == SALT and loaded.meta["policy"]
    dataset = dict((i, c) for i, c in cases)
    assert loaded.problems(dataset) == [] and loaded.warnings(dataset) == []
    assert loaded.ids("all") is None and loaded.ids("test") == loaded.test and loaded.ids("dev") == loaded.dev
    with pytest.raises(ValueError, match="unknown split"):
        loaded.ids("train")  # type: ignore[arg-type]
    assert sp.fingerprint(["b", "a"]) == sp.fingerprint(["a", "b"]) != sp.fingerprint(["a"])


def test_integrity_problems_are_all_detected(tmp_path):
    cases = items({"direct_kb": 10})
    dataset = dict(cases)
    good = doc_for(sp.plan(cases, salt=SALT))
    # overlap
    both = {**good, "dev": good["dev"] + [good["test"][0]]}
    assert any("both dev and test" in p for p in sp.load_split(write(tmp_path, both)).problems(dataset))
    # an id the dataset no longer has
    ghost = {**good, "dev": good["dev"] + ["D99"]}
    assert any("not in the dataset" in p for p in sp.load_split(write(tmp_path, ghost)).problems(dataset))
    # a dataset case nobody assigned
    assert any("without an assignment" in p and "D11" in p for p in sp.load_split(write(tmp_path, good)).problems({**dataset, "D11": "direct_kb"}))
    # a hand edit: an id moved from test to dev without refreshing the fingerprint
    moved = {**good, "test": good["test"][1:], "dev": good["dev"] + good["test"][:1]}
    assert any("fingerprint" in p for p in sp.load_split(write(tmp_path, moved)).problems(dataset))


def test_warnings_flag_lopsided_splits(tmp_path):
    cases = items({"direct_kb": 10})
    dataset = dict(cases)
    none_in_test = sp.load_split(write(tmp_path, doc_for({i: "dev" for i, _ in cases})))
    assert any("0 test case" in w for w in none_in_test.warnings(dataset)) and any("test share" in w for w in none_in_test.warnings(dataset))
    all_test = sp.load_split(write(tmp_path, doc_for({i: "test" for i, _ in cases}), "all.json"))
    assert any("10 test case" in w for w in all_test.warnings(dataset))


@pytest.mark.parametrize("content,match", [
    (None, "not found"), ("{not json", "malformed"), ('{"dev": []}', "malformed"), ('{"test": "D01", "dev": []}', "malformed"),
    ('{"test": ["A", "A"], "dev": []}', "twice"),
])
def test_load_split_rejects_missing_and_malformed_files(tmp_path, content, match):
    path = tmp_path / "s.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        sp.load_split(path)


def test_filter_cases_keeps_one_split(tmp_path):
    class C:
        def __init__(self, id):
            self.id = id
    cases = [C(f"D{i:02d}") for i in range(1, 11)]
    path = write(tmp_path, doc_for(sp.plan(items({"direct_kb": 10}), salt=SALT)))
    test_ids = {c.id for c in sp.filter_cases(cases, "test", path)}
    dev_ids = {c.id for c in sp.filter_cases(cases, "dev", path)}
    assert len(test_ids) == 3 and not test_ids & dev_ids and test_ids | dev_ids == {c.id for c in cases}
    assert len(sp.filter_cases(cases, "all", path)) == 10


# ------------------------------------------------------------------------------ the real, frozen split file
def _real():
    dataset = {c["id"]: c["category"] for c in json.loads((PROJECT_ROOT / "data" / "eval" / "dataset.json").read_text(encoding="utf-8"))["cases"]}
    return sp.load_split(PROJECT_ROOT / "data" / "eval" / "split.json"), dataset


def test_the_real_split_is_consistent_with_the_real_dataset():
    split, dataset = _real()
    assert split.problems(dataset) == [], "run: python scripts/freeze_split.py --check   (new cases need --add-new)"
    assert split.warnings(dataset) == []
    assert not split.test & split.dev and len(split.test) + len(split.dev) == len(dataset)


def test_the_frozen_test_cases_have_not_moved():
    """A test case may leave the test split only deliberately: list it in BURNED (with the reason) and in docs/FINDINGS.md."""
    split, _ = _real()
    missing = {i for i in FROZEN_TEST_IDS - split.test if i not in BURNED}
    assert not missing, f"frozen test cases moved out of the test split without being listed in BURNED: {sorted(missing)}"
    assert all(i in split.dev for i in BURNED), "a burned case must be in dev"


def test_every_large_category_and_the_adversarial_block_are_split():
    split, dataset = _real()
    by_stratum: dict[str, list[str]] = {}
    for cid, cat in dataset.items():
        by_stratum.setdefault(sp.stratum_of(cid, cat), []).append(cid)
    for name, ids in by_stratum.items():
        if len(ids) >= sp.MIN_STRATUM:
            assert any(i in split.test for i in ids) and any(i in split.dev for i in ids), name
    adversarial = [i for i in dataset if i.startswith("X")]
    assert 0.15 <= sum(1 for i in adversarial if i in split.test) / len(adversarial) <= 0.45


# ------------------------------------------------------------------------------ the runner
def write_run_split(tmp_path, test, dev):
    return write(tmp_path, doc_for({**{i: "test" for i in test}, **{i: "dev" for i in dev}}), "run_split.json")


def test_runner_split_selects_only_that_split_and_labels_the_run(make_runner, tmp_path):
    path = write_run_split(tmp_path, ["U1"], ["D1", "E1"])
    runner, agent, baseline, _ = make_runner(eval_split_path=path)

    test = runner.run(RunOptions(split="test", run_id="t1", use_judge=False))
    assert test["dataset"]["split"] == "test" and test["dataset"]["num_cases"] == 1 and test["progress"]["units_total"] == 2
    assert [c["question"] for c in agent.calls] == [UNANSWERABLE_Q] and baseline.calls == [UNANSWERABLE_Q]
    assert any("frozen TEST split" in n for n in test["notes"])
    assert json.loads((runner.run_dir("t1") / "config.json").read_text())["split_sha256"]

    dev = runner.run(RunOptions(split="dev", run_id="d1", use_judge=False))
    assert dev["dataset"]["num_cases"] == 2 and any("DEV split" in n for n in dev["notes"])

    everything = runner.run(RunOptions(run_id="a1", use_judge=False))
    assert everything["dataset"]["num_cases"] == 3 and everything["dataset"]["split"] == "all"
    assert not any("split" in n.lower() for n in everything["notes"])
    assert json.loads((runner.run_dir("a1") / "config.json").read_text())["split_sha256"] is None


def test_runner_refuses_a_missing_or_inconsistent_split_before_spending_anything(make_runner, tmp_path):
    runner, agent, baseline, _ = make_runner(eval_split_path=tmp_path / "nope.json")
    with pytest.raises(ValueError, match="not found"):
        runner.run(RunOptions(split="test", run_id="x"))
    incomplete = write_run_split(tmp_path, ["U1"], ["D1"])                            # E1 has no assignment
    runner2, agent2, _, _ = make_runner(eval_split_path=incomplete)
    with pytest.raises(ValueError, match="without an assignment"):
        runner2.run(RunOptions(split="dev", run_id="y"))
    assert not agent.calls and not agent2.calls and not baseline.calls
    runner2.run(RunOptions(run_id="z", mode="agentic", use_judge=False))              # 'all' does not need the split at all


def test_resume_refuses_a_split_file_that_changed(make_runner, tmp_path):
    path = write_run_split(tmp_path, ["U1"], ["D1", "E1"])
    runner, *_ = make_runner(eval_split_path=path)
    runner.run(RunOptions(split="dev", run_id="rs", use_judge=False))
    resumed, cases = runner.prepare(RunOptions(run_id="rs", resume=True))
    assert resumed.split == "dev" and len(cases) == 2                                   # the split is part of the saved options
    doc = json.loads(path.read_text())
    path.write_text(sp.dumps({**doc, "note": "edited"}), encoding="utf-8")              # same assignment, different file
    with pytest.raises(ValueError, match="split file changed"):
        runner.prepare(RunOptions(run_id="rs", resume=True))


def test_unknown_split_name_is_rejected():
    with pytest.raises(ValidationError):
        RunOptions(split="train")                                                        # type: ignore[arg-type]
    assert RunOptions().split == "all"


# ------------------------------------------------------------------------------ the command-line tool
def _freeze(tmp_path, *args):
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "freeze_split.py"), "--dataset", str(tmp_path / "ds.json"),
         "--split-file", str(tmp_path / "split.json"), *args],
        capture_output=True, text=True, cwd=tmp_path,
    )


def test_freeze_script_create_check_add_new_and_tamper_detection(tmp_path):
    def dataset(n):
        return {"cases": [{"id": f"D{i:02d}", "category": "direct_kb"} for i in range(1, n + 1)]}

    (tmp_path / "ds.json").write_text(json.dumps(dataset(10)), encoding="utf-8")
    assert _freeze(tmp_path, "--check").returncode == 2                                # no split yet
    created = _freeze(tmp_path, "--create")
    assert created.returncode == 0 and "3 test / 7 dev" in created.stdout
    assert _freeze(tmp_path, "--create").returncode == 2                               # frozen: no overwrite
    assert _freeze(tmp_path, "--check").returncode == 0
    before = json.loads((tmp_path / "split.json").read_text())

    (tmp_path / "ds.json").write_text(json.dumps(dataset(25)), encoding="utf-8")       # 15 new cases
    assert _freeze(tmp_path, "--check").returncode == 1                                # unassigned cases are a problem
    added = _freeze(tmp_path, "--add-new")
    assert added.returncode == 0 and "15 new case" in added.stdout
    after = json.loads((tmp_path / "split.json").read_text())
    assert set(before["test"]) <= set(after["test"]) and set(before["dev"]) <= set(after["dev"])
    assert len(after["test"]) == sp.quota(25) and _freeze(tmp_path, "--check").returncode == 0
    assert "Nothing to add" in _freeze(tmp_path, "--add-new").stdout

    after["test"].pop(0)                                                                # a hand edit
    (tmp_path / "split.json").write_text(json.dumps(after), encoding="utf-8")
    assert _freeze(tmp_path, "--check").returncode == 1
