# Development / test split of the evaluation dataset

Prompts, thresholds and routing rules in this project were tuned while looking at evaluation failures (for
example the routing leak, the abstention wording and the community-page rule). A score measured on the very
questions that drove those changes is optimistic. The split gives the same dataset two honest uses.

| Split | Use it to | Do not |
|---|---|---|
| **dev** (65 cases) | look at failures, change code, prompts or thresholds, re-run as often as you like | quote it as the headline result |
| **test** (28 cases) | **report** a number | change the system because of an individual failure |

If a test failure does make you change something, the case is *burned*: move it from `test` to `dev` in
`data/eval/split.json`, run `python scripts/freeze_split.py --refresh-fingerprint`, add the case id and the reason to
`BURNED` in `tests/unit/test_eval_split.py`, and write one line in `docs/FINDINGS.md`. The unit test
`test_the_frozen_test_cases_have_not_moved` fails until you do, so a case cannot drift out of the test set silently.

## How the split was made

* `data/eval/split.json` was created once, on 2026-11-21, by `scripts/freeze_split.py --create` (rule and policy are
  stored in the file).
* Stratified by category, with the adversarial block (`X01`-`X20`) as its own stratum, so every category with at
  least three cases has both dev and test questions. About 30% of each stratum is test; a stratum of one or two cases
  stays in dev.
* Inside a stratum the order is a salted hash of the case id, not anybody's opinion of which questions are easy.
* **Frozen:** existing assignments are never recomputed. Cases added to the dataset later get an assignment with
  `python scripts/freeze_split.py --add-new`, which leaves every old assignment untouched. A stored fingerprint of the
  test ids catches a hand edit, and `scripts/run_eval.py --validate-only` checks the file against the dataset.

```powershell
python scripts\freeze_split.py --check        # integrity against the dataset (no network, no cost)
python scripts\freeze_split.py --show         # cases per category in dev and test
python scripts\freeze_split.py --add-new      # after adding questions to the dataset
python scripts\run_eval.py --split dev --per-category 3      # tune here
python scripts\run_eval.py --split test                      # report this
python scripts\calibrate_retrieval.py --split dev            # threshold tuning stays on dev
```

The dashboard has the same choice (**Admin → Evaluation → Question set**). `run_eval.py --split` writes the split name into the run's `summary.json` and notes; a test-split run says so in its
notes. `calibrate_retrieval.py`, `eval_rerank_ab.py` and `diagnose_rerank.py` default to `all` (their published numbers
were computed on all cases), so pass `--split dev` when tuning from now on.

## What this does NOT give you

* **The test split is less contaminated, not unseen.** All 93 questions existed and were visible while the system was
  developed, and the earlier reranker and calibration studies used all of them. The adversarial block was written after
  the routing rules, but after its authors had seen what failed. The honest way to get a clean test set is to write
  new questions *after* the freeze, without looking at system output, add them with `--add-new`, and report on those.
  Twenty to thirty fresh questions written that way are worth more than the current 28.
  A block of 40 now exists as a separate file, [FRESH_QUESTIONS.md](FRESH_QUESTIONS.md); it has not been run yet.
* 28 questions is small. One question is 3.6 points of the test split; read the intervals (`docs/ABLATION.md`) before
  concluding anything from a difference.
* The split protects against *human* tuning on test cases. It does not control LLM sampling noise between runs.
