# Hybrid retrieval experiment: does BM25 + dense (RRF) beat dense alone?

> **Status: run once (2026-11-22). Outcome under the pre-registered rule: `evidence_of_harm`. Hybrid retrieval is not adopted
> and nothing in the application uses BM25.** Every section except "Result" was written before the first run and is unchanged
> apart from this note; the hypothesis, the fixed parameters and the decision rule are not edited after a result is seen, and a
> change is a new experiment (see "Tuning policy").

## Why look at this at all

The retrieval regression check (`data/eval/retrieval_baseline.json`) says dense retrieval with MiniLM already places the
expected document near the top for most of the 59 questions that name expected documents. That leaves little room for any
retrieval change to help, so the honest expectation is a small or undetectable effect. This is an expectation, not a result.
The experiment exists to replace the expectation with a measurement, and to show the method: fix the rule first, run
once, report whatever comes out.

Lexical matching is a classic complement to embeddings for IT text: product names, error codes and acronyms (`VPN`,
`MFA`, `0x80070005`) are rare tokens that a small embedding model can blur. Whether that matters on this corpus is
exactly what is unknown.

## The three arms (retrieval only, no LLM, no Tavily, zero tokens)

| Arm | What it is |
|---|---|
| `dense` | The application's own retrieval for the original question (`rank_case`, the call the evaluation runner and the regression check use). Reranker and per-document cap forced off. |
| `bm25` | BM25 over every chunk the question's role may read, in pure Python (`vectorstore/lexical.py`). |
| `hybrid` | The two ranked lists fused with Reciprocal Rank Fusion. |

* **Same text on both sides.** The BM25 index is built from the text that is embedded: `title | section` then the chunk text.
* **Access control comes first.** The BM25 arm filters by the role's access levels **before** ranking, like Pinecone's
  metadata filter, so a restricted chunk cannot reach an employee through either list. A unit test checks this for all three
  arms, and checks that an admin does get the restricted document (so the test is not vacuous).
* **The corpus is read from the live index** (`PineconeStore.iter_chunks`: list ids, then fetch). The script prints which
  documents and how many chunks it built the lexical index from, and warns if they differ from `data/kb_documents`.
* **The dense arm must reproduce the committed baseline.** When the setup matches the baseline's recorded context, the script
  reports the largest difference between its dense means and `retrieval_baseline.json` (expected ≈ 0). If it is not ≈ 0, the
  index differs and the comparison is not trusted.

## Fixed parameters (chosen before any run; not tuned)

| Parameter | Value | Note |
|---|---|---|
| BM25 `k1` | 1.5 | common default range 1.2 to 2.0 |
| BM25 `b` | 0.75 | common default |
| BM25 IDF | Lucene form, `ln(1 + (N - df + 0.5) / (df + 0.5))` | never negative |
| Tokeniser | lower-case, alphanumeric, stop words removed, digits kept, Harman S-stemmer (plurals only) | no external NLP library |
| RRF `k` | 60 | value from the original RRF paper; equal weights |
| Pool | `EVAL_RETRIEVAL_POOL` (20) | the same pool the other retrieval measurements use |

## What decides the outcome

Five **document-level** metrics, compared as paired differences (hybrid minus dense) over the same questions, with a
percentile bootstrap interval (`EVAL_BOOTSTRAP_RESAMPLES`, `EVAL_BOOTSTRAP_SEED`, `EVAL_CONFIDENCE`):

`doc_recall@1`, `doc_recall@3`, `doc_recall@5`, `doc_mrr`, `doc_ndcg@5`.

A metric is `better` or `worse` only if its interval excludes zero **and** at least 10 questions contribute (the same rule as
the ablation study).

| Outcome | Rule |
|---|---|
| `evidence_of_harm` | any document metric is `worse` |
| `evidence_of_benefit` | no document metric is `worse` and at least 2 of the 5 are `better` |
| `no_evidence_of_benefit` | anything else |

Why 2 of 5: the five metrics are strongly correlated, and testing several of them gives one lucky result too many chances;
requiring two keeps one noisy metric from deciding. They are correlated, so this is a guard, not a multiple-testing correction.

**`no_evidence_of_benefit` is not evidence that the arms are equal.** With 59 questions and dense retrieval already strong, the
test has little power to see a small gain. Even `evidence_of_benefit` only licenses a follow-up (a production design and
variants tuned on dev, judged once on test); it does not adopt anything.

**Chunk-level metrics are reported but never decide.** A chunk counts as relevant when it contains the keywords of an expected
answer point, which favours lexical matching: BM25 would be graded by the same words it matches on.

## Which questions

All 59 questions that name expected documents, dev and test together. That is allowed because **nothing is tuned**: no
parameter was chosen by looking at these questions, so the frozen test split is not contaminated. The dev and test rows
are still printed separately, for transparency, not for selection.

## Tuning policy

Changing `k1`, `b`, `k`, the weights, the tokeniser, or adding query expansion creates a **new experiment** with a new
pre-registration: tune on the dev split only, then judge once on the test split. A result from this experiment must not be
used to pick the variant that is then judged on the same questions.

## What it does not tell you

* **Not answer quality.** It measures which documents and chunks are retrieved, not whether answers become more correct or
  more faithful. That needs a Groq-token run of the evaluation with the new retriever behind the graph.
* **Small, synthetic corpus.** 11 short documents. BM25's IDF statistics on so small a corpus are noisy, and the result may
  not transfer to a larger or real one.
* **Intervals describe the question set, not run-to-run noise.** All three arms are deterministic.
* **Latency and index maintenance are not measured.** A production hybrid needs a lexical index built at ingest, kept in
  sync on delete and re-index, and queried with the access filter inside the query. None of that exists; the experiment
  builds the index in memory from a full listing of the store.

## Result (run 2026-11-22)

**Outcome under the pre-registered rule: `evidence_of_harm`**: `doc_ndcg@5` is worse with an interval that excludes zero.
Hybrid retrieval is **not adopted**; the application is unchanged. The arms are deterministic: two full runs
(`hybrid_ab_20261122T000016` and `hybrid_ab_20261122T000155`) wrote byte-identical `per_case.csv` files.

Setup check: the lexical index was built from 63 chunks in 11 documents, matching `data/kb_documents`, and the dense arm
reproduced the committed retrieval baseline (largest difference 0.00005, which is the baseline's rounding to four decimals).

Hybrid minus dense, all 59 questions that name expected documents (paired bootstrap, 95% interval):

| Document metric | Dense | Hybrid | Difference [interval] | Questions better / worse | Verdict |
|---|---|---|---|---|---|
| doc_recall@1 | 0.822 | 0.805 | -0.017 [-0.085, +0.051] | 2 / 3 | not distinguishable |
| doc_recall@3 | 0.975 | 0.966 | -0.008 [-0.025, +0.000] | 0 / 1 | not distinguishable |
| doc_recall@5 | 0.983 | 0.983 | +0.000 [+0.000, +0.000] | 0 / 0 | not distinguishable |
| doc_mrr | 0.975 | 0.963 | -0.011 [-0.054, +0.025] | 2 / 3 | not distinguishable |
| doc_ndcg@5 | 0.885 | 0.825 | -0.061 [-0.104, -0.021] | 9 / 23 | **worse** |

* **No document metric was `better` in any subset.** Dev (40 questions): recall@1 +0.025 and MRR +0.013, neither distinguishable
  from zero; `doc_ndcg@5` 0.871 to 0.817, worse. Test (19 questions that have expected documents): recall@1 0.868 to 0.763 and
  MRR 1.000 to 0.939, neither distinguishable; `doc_ndcg@5` 0.916 to 0.840, worse. The rows are shown for transparency; nothing
  was selected on them.
* **BM25 alone** (informational, not part of the rule) is worse than dense on recall@1 (0.636 vs 0.822), recall@3 (0.831 vs
  0.975), MRR (0.844 vs 0.975) and `doc_ndcg@5` (0.631 vs 0.885); recall@5 (0.941 vs 0.983) is not distinguishable.
* **Hybrid changed the document MRR of only 5 questions**: improved on R09 and R10, worsened on A02, R05 and R07.

### How to read it

* **What `doc_ndcg@5` measures here.** It ranks chunks and counts a chunk as relevant when it belongs to an expected
  document, so it rewards filling the top five slots with chunks of the expected documents. It does not ask whether the right
  document was found at all; that is recall and MRR. Hybrid left recall@5 unchanged at 0.983 and let off-document chunks into
  the top five.
* **One metric out of five triggered the verdict, and the five are correlated.** This is not five independent findings. The rule
  said in advance that any `worse` document metric means harm, so the verdict is reported as the rule gives it, not softened.
* **Exploratory and post hoc (chosen after seeing the numbers; not part of the decision).** The four largest drops in
  `doc_ndcg@5` are all `needs_rewriting` questions (R03 -0.53, R07 -0.52, R08 -0.49, R01 -0.38). The mean difference is -0.181 over
  the 10 `needs_rewriting` questions and -0.013 over the 31 `direct_kb` questions. A possible explanation, **not tested**: these
  questions use different words from the documents, so BM25 adds lexically matching but off-topic chunks. Ten questions can
  only suggest a hypothesis.
* **The corpus caveat cuts both ways.** Dense retrieval is already near its ceiling on 11 short documents, so there is little for
  BM25 to add and a lot for it to dilute. The result may not transfer to a larger corpus with many error codes and product names.

### What follows

* Nothing is tuned on this result. A variant (for example dense-weighted fusion, or BM25 only when dense retrieval looks weak)
  would be a **new** pre-registered experiment: tune on dev, judge once on test.
* The test split's retrieval rows were looked at here (as pre-registered, all 59 questions), so a later test-split judgement of a
  variant is less clean for retrieval than a fresh block of questions written after this result would be.
* Answer quality with a different retriever was never measured and is not claimed.

## Run it

```powershell
python scripts\eval_hybrid_ab.py --corpus-only    # list the chunks the lexical index would be built from, then stop
python scripts\eval_hybrid_ab.py                  # run the experiment (needs PINECONE_API_KEY and the ingested KB; no Groq, no Tavily)
python -m pytest tests\unit\test_lexical.py tests\unit\test_hybrid_ab.py -q
```

Output: printed tables, the pre-registered decision, and `data/eval/results/hybrid_ab_<timestamp>/{summary.json,per_case.csv}`.

## Files

| File | Responsibility |
|---|---|
| `src/copilot/vectorstore/lexical.py` | tokeniser, BM25 index, Reciprocal Rank Fusion (stdlib only) |
| `src/copilot/vectorstore/pinecone_store.py` | `iter_chunks`: the full listing used to build the lexical index (not on the request path, not access-filtered) |
| `src/copilot/evaluation/hybrid_ab.py` | corpus, the three arms, paired summary, the decision rule |
| `scripts/eval_hybrid_ab.py` | command-line runner |
| `tests/unit/test_lexical.py`, `tests/unit/test_hybrid_ab.py` | unit tests (fake index and fake embedder, no network) |
