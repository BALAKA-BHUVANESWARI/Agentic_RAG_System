# Fresh held-out questions (F01 to F40)

Every number measured so far comes from questions that were visible while the system was being built. The frozen
dev/test split ([EVAL_SPLIT.md](EVAL_SPLIT.md)) makes that less bad but not clean, and says so. This block is the
cleaner judge that document asked for: **40 questions written on 2026-11-22, after the split was frozen, without running
the system on any of them and without looking at its output for them.** The labels refer to the synthetic demo knowledge base
only; nothing here is real company policy.

## What is in it

| Category | Questions | Notes |
|---|---:|---|
| direct_kb | 10 | different facts from D01-D12 where the documents allow it; two use the `admin` role (F10, F16) |
| paraphrased | 6 | |
| needs_rewriting | 5 | vague or symptom-style wording (F17 and F18 deliberately sit next to a document about a different problem) |
| insufficient_private | 4 | plausible IT questions the documents do not answer; must abstain and must not search the web |
| current_external | 3 | stable public vendor facts (Windows 11 requirements, Teams ports, Windows 10 end of support) |
| conflicting | 3 | the current and the superseded password policy; the labels expect the older edition to be stated (F29-F31). Since the older policy is marked superseded these labels do not measure conflict handling: results are also reported without this category ([finding 10](FINDINGS.md)) |
| unanswerable | 5 | undocumented or restricted detail (F32 names a keyword from the restricted runbook as a leak check) |
| out_of_scope | 4 | |

The file is `data/eval/fresh_dataset.json` (same schema as `dataset.json`). It is a separate file on purpose: the main
dataset's size is pinned by other tests, and a separate file cannot be tuned on by accident.

## How honest is "fresh"?

* **Same author as the system.** The questions were written by the assistant that also helped build and analyse the
  system, with the knowledge base and the main dataset open. They are not written by a neutral third party.
* **Known failure patterns were in my head.** Some questions echo patterns found earlier: F20 (sharing a login, like R10),
  F23 and F32 (restricted detail, like U05 and U04), F17 and F18 (a symptom-style question next to a similar
  document). Nothing in the system's output for these exact questions was seen.
* **Small.** 40 questions give wide intervals (about plus or minus 15 points on a rate near 50 to 90 percent). A difference of
  a few points between baseline and agentic will not be distinguishable; say "not distinguishable", not "no difference".
* **External cases depend on the live web.** F26 to F28 are stable public facts, but whether they are answered depends on
  the search results of that day. One run is an observation.

## Rules

1. **Report only.** Do not change prompts, thresholds, routing or labels because of an individual failure here. If you do,
   say so in `docs/FINDINGS.md` and treat the changed case as burned (it can no longer judge the change you made for it).
2. **A fingerprint guards the labels** (`data/eval/fresh_dataset.sha256`, checked by `tests/unit/test_eval_fresh_dataset.py`).
   A label typo found *before the first run* may be fixed: regenerate the fingerprint and write one line below. After the first
   run, any edit must be recorded below with the reason.
3. **What will be reported, decided now:** overall correct, answerable-correct, correct abstention, hallucinated (should
   abstain), unnecessary external searches and citation correctness, for baseline and agentic, each with its 95%
   interval and the paired difference. No metric is added after seeing the result.

## How to run it

The runner reads the dataset path from the environment. The real `split.json` does not describe these questions, so point the
split path at a file that does not exist. In PowerShell, from `it-support-copilot/`:

```powershell
$env:EVAL_DATASET_PATH = "data\eval\fresh_dataset.json"
$env:EVAL_SPLIT_PATH = "data\eval\no_split_for_fresh_block.json"
python scripts\run_eval.py --validate-only        # expect: 40 cases, "No split file at ...", "Dataset is valid."
python scripts\run_eval.py --no-judge --delay 5   # spends Groq and Tavily tokens
Remove-Item Env:EVAL_DATASET_PATH, Env:EVAL_SPLIT_PATH
```

Cost: extrapolating from the 2026-11-22 attacks run (about 5,300 tokens per question for baseline and agentic together, judge
off), the 40 questions need roughly 210,000 tokens, slightly more than one day of the free tier (200,000). The circuit breaker will
stop the run when the quota is gone; continue it the next day with `python scripts\run_eval.py --resume <run_id>` (with the same two
environment variables set). Keep exactly the 11 core documents in the index and do not run other Groq work around it.

## Edit log

* 2026-11-22: created. Fingerprint `931aeabfa903552fcd542935016e69d05d241581d359ef8dde8e17e3c01c171e`. Not yet run.
* 2026-11-22 (later, before any run): documentation only, no label or question changed and the fingerprint is unchanged. The `conflicting` row above gained a caveat: F29-F31 carry the same expectation as C01-C06 (the older edition of the password policy stated as a live conflict), which no longer fits a corpus where that edition is marked superseded. They stay as written; the renderer reports overall correct with and without the category ([finding 10](FINDINGS.md)).
