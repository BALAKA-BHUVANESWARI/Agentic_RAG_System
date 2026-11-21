# Ablation study: what does each agent step buy?

> **Status: the machinery is built and unit-tested; no ablation results exist yet.** A run needs Groq tokens, so it is
> scheduled for a separate day. Until a run is on disk, nothing in this repository claims that any step helps.

The README compares a plain RAG baseline with the full agent. That comparison cannot say *which* step of the agent
earns its extra latency and tokens, so this study adds the steps one at a time.

## The ladder

Every rung is the **same LangGraph workflow** with settings overrides (there is no second implementation of the
graph). The model, corpus, retriever, prompts and dataset are shared by all rungs.

| Rung | Adds | Settings overrides relative to the real application |
|---|---|---|
| `baseline` | nothing: retrieve, then generate with citations | (separate `BaselineRAG` class, no graph) |
| `classify` | query classifier; out-of-scope questions skip retrieval | `AGENT_USE_GRADER=false`, `MAX_REWRITE_RETRIES=0`, `EXTERNAL_SEARCH_ENABLED=false`, `AGENT_USE_VALIDATOR=false` |
| `grade` | LLM evidence grader (weak evidence abstains) | `MAX_REWRITE_RETRIES=0`, `EXTERNAL_SEARCH_ENABLED=false`, `AGENT_USE_VALIDATOR=false` |
| `rewrite` | query rewriter and bounded retrieval retries | `EXTERNAL_SEARCH_ENABLED=false`, `AGENT_USE_VALIDATOR=false` |
| `external` | graded Tavily fallback under the private-first rules | `AGENT_USE_VALIDATOR=false` |
| `agentic` | final grounding validator: **the real application** | none |

`python scripts/run_eval.py --list-variants` prints this table from the code, so it cannot drift from it. A unit test
checks that each rung lifts exactly one restriction of the previous one and that every override names a real setting.

What the two new switches do (they default to `true`; leave them alone outside an evaluation):

* `AGENT_USE_GRADER=false`: the grader node makes no LLM call and treats every retrieved chunk as sufficient evidence.
  Nothing retrieved still abstains. The trace event says `grader skipped (ablation)` and the stored grade carries
  `skipped: true`, so a skipped grader is **not** scored as a grading decision (`evidence_grading_accuracy` ignores it).
* `AGENT_USE_VALIDATOR=false`: the final LLM grounding check is skipped and the audit record says so
  (`validation.skipped`). Citation markers are still resolved in code by the answer generator, and an abstention
  passes through unchanged.

Neither switch can weaken a routing rule: with the grader off there is no "insufficient" verdict, so web search can
never be reached from that rung (tested in `tests/workflow/test_ablation.py`).

## Running it

```powershell
python scripts\run_eval.py --list-variants
# Cheapest useful first look: 8 questions (one per category) on three rungs, no judge
python scripts\run_eval.py --variants baseline,grade,agentic --per-category 1 --no-judge --run-id abl-smoke
# Everything; resumable like any other run, and the circuit breaker still stops it on repeated provider errors
python scripts\run_eval.py --variants all --run-id ablation-1
python scripts\run_eval.py --resume ablation-1
```

Token cost per rung has not been measured for this dataset. Read the `total tokens` row after the smoke run before
scaling up, and do not change the model between rungs or between days of one run (a comparison across models measures
the model).

Output: the usual run directory, plus `summary.json -> ablation` (one row per rung, and for every step the movement of each
metric with a better / worse / tie verdict), `summary.json -> variants` (the exact overrides used) and `ablation.*`
scopes in `metrics.csv`. The terminal summary prints one column per rung and a short findings list built from the
numbers only.

## How to read the result (and what not to claim)

* The step table's **tie / better / worse** uses a quick display rule (a rate moved by less than 2 points, or a count
  or average by less than 5%). It is not a significance test. The evidence is the `intervals` block that every run now
  writes: for each step, the questions that flipped (`gained` / `lost`), the difference in points and a 95%
  paired-bootstrap interval, and a verdict that says `better` or `worse` only when the interval excludes zero and at
  least 10 questions contribute (otherwise `not distinguishable` or `too few questions`). The terminal summary prints it.
* Those intervals describe which questions happened to be in the dataset. They do **not** cover the LLM giving a
  different answer next time, so one run per rung is still one sample; repeat the key rungs before quoting a step.
* `classify` changes two things relative to the baseline (the out-of-scope shortcut and the agent's own answer
  path), so its step is not a pure measure of the classifier.
* The `external` rung is only worth anything on the `current_external` cases (8 of 93); a flat result elsewhere is
  expected.
* The adversarial block (X01-X20) was written after the agent's rules were tuned on the earlier cases. It is useful
  for spotting regressions, not as an unbiased held-out test.
* Report negative steps as honestly as positive ones. If a rung costs latency and tokens and buys nothing on this
  dataset, that is a finding (the reranker experiment in `docs/FINDINGS.md` is the precedent).
