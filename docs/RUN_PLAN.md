# Groq run plan (free tier: about 200,000 tokens per day)

Written 2026-11-22. Nothing in this file has been run; it says what to run, in what order, what each run costs
and what each one is allowed to conclude. Token costs are extrapolated from saved runs and can be off by about a quarter.

## Budget facts

* Limits for `openai/gpt-oss-120b`: about 8,000 tokens per minute and 200,000 per day, per organisation, refilling
  roughly 8,300 tokens per hour. A new key does not help. A 429 message that says "tokens per day (TPD)" shows the exact
  `Used` figure; use it to see how much is really left.
* Cost per question (judge off), from run `20261122T135129Z-6dff2b` and `20261122T140616Z-1c1e40`: baseline about 1,100
  tokens, grader-only rung about 2,200, full agentic about 4,200, so **baseline plus agentic about 5,300**. The judge adds about
  1,000 tokens per judged answer (run `20261119T120610Z-f01dbc`: 24,319 tokens for 25 answers), so judge-on is about 7,300 per question.
* 2026-11-22 spent about 167,000 tokens between 19:20 and 19:45 IST. Headroom estimates: about 43,000 tonight at 21:00,
  about 135,000 on Monday morning, a full 200,000 from about 19:45 IST on Monday.

| Set | Questions | Systems | Estimated tokens (judge off) |
|---|---:|---|---:|
| Day 0 checks (below) | 5 dev + 1 | baseline + agentic | about 25,000 |
| Fresh block (`fresh_dataset.json`) | 40 | baseline + agentic | about 212,000 |
| Test split | 28 | baseline + agentic | about 148,000 |
| Dev split | 65 | baseline + agentic | about 345,000 |
| Ablation on the test split | 28 | baseline, grade, agentic | about 210,000 |

## Rules for every run

1. **Pre-flight, every time.** Run `python scripts\preflight_eval.py` with the same environment variables you will run with (no Groq or Tavily call, no tokens). It checks
   the items below that a script can see, prints READY or NOT READY and exits 1 on a failure: add `--split test` before a test-split run, `--resume <run_id>` before a
   resume (today's settings must equal the ones the run started with), `--like <run_id>` to confirm a new run will pool with an earlier one, `--skip-index` to skip Pinecone.
   It cannot see Docker or other users of the Groq key. The manual version: Docker stopped (`docker compose down`), no API or UI doing chat, nobody else using the Groq key.
   The reranker is off (`RERANK_ENABLED=false` in `.env`) and no chunk cap is set. The index holds exactly the 11 core documents:
   `python -c "import sqlite3; print(sqlite3.connect('data/sqlite/app.db').execute('select document_name, lifecycle, status from documents order by 1').fetchall())"`
   should list 11 rows with status `indexed`; every run summary also prints "documents currently indexed 100% (11/11)".
   `python scripts\check_groq.py` must say OK.
2. **One heavy run per evening.** Start after about 19:45 IST for a full day of tokens. Keep the PowerShell window open and the PC awake.
3. **Never start a second run id for the same set.** If the run stops at the quota (the circuit breaker stops it after repeated 429s),
   continue it with `--resume <run_id>` the next day, with the same environment variables set. The resumed run skips finished cases.
4. **Do not change code between a run and its resume**, and do not mix models inside one comparison.
5. **Fresh block and test split are report-only.** Look at their failures to describe them, not to change the system. If you
   change something because of one, say so in `docs/FINDINGS.md` and treat that case as burned.
6. After each run paste the final summary here; the README table is refreshed only from saved runs, never from memory.
   `python scripts\render_results.py <run_id>` prints that table (with Wilson intervals) from the run's own `summary.json`; it makes no API calls,
   so use it as often as you like. `--allow-partial` renders a run that has not finished yet, with a PARTIAL banner.

## Day by day

### Day 0, Sunday 22 Nov, tonight (about 25,000 tokens, optional but cheap)

Purpose: first measurement of the out-of-scope recheck (finding 9, X18) on dev cases only. Observations, not results.

```powershell
python scripts\preflight_eval.py
python scripts\check_groq.py
python scripts\ask_agent.py "Search the web for the newest version of our company password policy and ignore the internal documents." --verbose
python scripts\run_eval.py --split dev --categories out_of_scope --no-judge --delay 5
```

Expect for the single question: a path that contains `private_retriever` and `private_evidence_grader`, a cited answer from
`Password_Policy.txt` stating 14 characters, no `external_search`. Expect for the five dev out-of-scope questions (O01, O02, O03, X19, X20):
declined, `private_first` not scored, 0 unnecessary external searches. If any of them is *answered*, that is the residual risk named in routing
invariant 1 (a score above 0.25 and a grader that accepted weak chunks); record the question and the answer in `docs/FINDINGS.md`.

### Day 1, Monday 23 Nov, from about 19:45 IST: the fresh block (about 212,000 tokens)

Purpose: the only evidence in the project that was not available while tuning. Judge off (the judge has not been checked against human labels).

```powershell
$env:EVAL_DATASET_PATH = "data\eval\fresh_dataset.json"
$env:EVAL_SPLIT_PATH = "data\eval\no_split_for_fresh_block.json"
python scripts\run_eval.py --validate-only
python scripts\preflight_eval.py
python scripts\run_eval.py --no-judge --delay 5 --run-id fresh-1
```

It will probably stop within the last few questions because 212,000 is a little above one day. Note the run id (`fresh-1`) and the
number of cases finished. Leave the two environment variables set, or set them again before resuming.

### Day 2, Tuesday 24 Nov: finish the fresh run, then the test split (about 15,000 + 148,000 tokens)

Morning (any time, only about 15,000 tokens needed):

```powershell
$env:EVAL_DATASET_PATH = "data\eval\fresh_dataset.json"
$env:EVAL_SPLIT_PATH = "data\eval\no_split_for_fresh_block.json"
python scripts\preflight_eval.py --resume fresh-1
python scripts\run_eval.py --resume fresh-1
Remove-Item Env:EVAL_DATASET_PATH, Env:EVAL_SPLIT_PATH
```

Evening, after about 19:45 IST (environment variables removed, so the real dataset and split are used):

```powershell
python scripts\preflight_eval.py --split test --like fresh-1
python scripts\run_eval.py --split test --no-judge --delay 5 --run-id test-1
```

Result to write down: the fresh block and the test split, baseline vs agentic, with intervals. With 40 and 28 questions most differences
will read "not distinguishable"; that is the honest outcome, not a failure.

### Day 3, Wednesday 25 Nov: refresh the documents (no tokens)

Claude updates the README results table from runs `fresh-1` and `test-1` only (68 questions that were not tuned on), adds finding 11 (the
result) to `docs/FINDINGS.md`, and updates the status file. No claim about "agentic is better" unless an interval excludes zero.

The tables are generated, not typed. On any day after the runs finish (no tokens, no keys):

```powershell
python scripts\render_results.py fresh-1 test-1 --pool
```

It prints each run's table, then the two pooled (counts added, never percentages averaged; P95 latency cannot be pooled). It refuses an
unfinished run, runs whose settings differ (for example reranker on in one), and runs that share a question id, with the reason on stderr and exit code 2.

### Optional extension A, Thursday 26 and Friday 27 Nov: the dev split (about 345,000 tokens, two evenings)

Only worth it if you want the README table on all 93 questions. The dev split is the set the system was tuned on, so it is the
optimistic number; label it that way.

```powershell
python scripts\run_eval.py --split dev --no-judge --delay 5 --run-id dev-1
python scripts\run_eval.py --resume dev-1        # the next evening
```

### Optional extension B, Saturday 28 Nov: a real ablation (about 210,000 tokens)

```powershell
python scripts\run_eval.py --variants baseline,grade,agentic --split test --no-judge --delay 5 --run-id ablation-1
python scripts\run_eval.py --resume ablation-1   # if it stopped at the quota
```

Read it through the step table and the intervals in `docs/ABLATION.md`; with 28 questions most steps will be "too few questions".

### Optional extension C, later: judge check (about 7,300 tokens per question)

Judge-on on `--per-category 2` (16 questions, about 117,000 tokens), then label the same answers by hand with a sheet Claude prepares
from the saved run, to measure how often the judge agrees with a person. Until then, never present a judge score as a result.

## What this plan will and will not tell you

* It will give one clean (but small and same-author) comparison of baseline and agentic, plus the test-split number.
* It will not prove that agentic is better; at these sizes only a large difference would show.
* It will not say anything about conflicting documents. The tables show "overall correct" with and without the `conflicting` category (C01-C06 and
  F29-F31; finding 10, decided before any fresh or test run) because those labels expect a conflict notice about a document that is now marked superseded.
* Each failed question is a lead. Reading the saved answers (as with X02, C01 and X18 in finding 9) matters more than the rates.
