# Findings: what testing this system actually turned up

A record of the problems found while building and testing the IT Support Copilot, how each was found, what caused it, and how it was fixed or left open. Every number below comes from a recorded run, a saved result file or the SQLite audit log, and each entry names its source. Documents are synthetic demo documents, so these findings say more about the engineering than about any real company's data.

Short version: several defects were found by measurement (an evaluation metric, the system-health numbers, the audit log or a live demo) rather than by reading code, one experiment (a reranker) gave a negative result that was kept, and several claims are still **not** proven. The last section lists those.

---

## 1. The evaluation caught a routing leak (unnecessary web searches)

- **Symptom.** In the first complete 120b comparison run (`20261118T062347Z-eb71dc`), the question "What is the process to request a second monitor for my desk?" triggered a Tavily search. It is company-specific and undocumented, so the rules say it must abstain without any web search. The metric "unnecessary external searches" read 1 of 21.
- **How it was found.** Not by a code review: the agent-workflow metrics in the evaluation report counted it. A similar case (I03) had shown the same thing on a smaller model.
- **Cause.** The audit record of one request (`11d32d056f5a43ac`) showed the classifier had correctly said `internal_it` with 0.99 confidence. On the third grading pass the evidence grader returned `recommended_action = external_search`, and the routing function accepted that recommendation as an alternative to the classifier's finding. A grader's opinion alone could open web search.
- **Fix.** `external_eligible` in `src/copilot/graph/routing.py` now requires the classifier's intent (`external_current` or `both`); the grader's recommendation is only a signal in the trace. A regression test replays the case.
- **Re-measured.** In the next run (`20261119T120610Z-f01dbc`) unnecessary external searches were 0 of 21, and the live demo repeated the check (S2: abstained, `tavily_calls = 0`).
- **Lesson.** A routing rule that is "obviously right" in a design document can be bypassed by one condition written as `A or B`. The metric that exists to catch it is what found it.

## 2. Ingestion success was 7 of 11, and the system metrics said so

- **Symptom.** The system section of the first evaluation reported ingestion success of 7/11 while the knowledge base held 11 documents.
- **Cause.** Pinecone SDK 10.x returns `ListItem` objects from `index.list()`; `PineconeStore.delete_document` treated them as strings. It crashed whenever old vectors existed, that is on every re-index, version change or delete. Four documents were left `failed`.
- **Fix.** `delete_document` accepts the SDK objects; the test `test_delete_document_accepts_sdk_list_items` pins it. After re-ingesting, SQLite showed 11 of 11 indexed.
- **Lesson.** Tests with a fake vector store had passed the whole time. The real SDK differed from the fake. This is why the project also has live integration tests and a system-health section in the evaluation.

## 3. Provider quota must not look like a wrong answer

- **Symptom.** A 24-question run (`20261117T092125Z-d593c8`) hit Groq's free-tier daily token cap. From the 13th question on, every call returned HTTP 429, and a later run (`20261120T125742Z-0d4509`) stopped at 26 of 48 for the same reason.
- **Risk.** Scoring those failures as abstentions would have inflated the "correct abstention" rate.
- **Fix.** A failed infrastructure call is recorded as an `error`, never as an abstention; a circuit breaker (`EVAL_MAX_CONSECUTIVE_ERRORS`, default 3) stops the run after repeated failures and the run can be resumed (`--resume <run_id>`). The dashboard shows a red "stopped" banner and does not score the missing questions.
- **Lesson.** The partial numbers of a stopped run are never quoted anywhere in this repository.

## 4. A reranker that made retrieval worse, and the attempts to explain it

- **Setup.** An optional cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) can reorder the retrieved chunks. It is implemented and **off by default**.
- **First finding (a bug in my own code).** The reranker scored the bare chunk text, while the embedder had seen `title | section` plus the text. Chunk MRR fell from 0.932 to 0.822 (32 questions). Scoring the same context fixed it (0.930).
- **Second finding (still negative).** On 47 questions that have expected documents, chunk MRR is 0.913 without the reranker and 0.831 with it (`scripts/diagnose_rerank.py`, run `rerank_diag_20261121T220154`). It hurt 10 questions and helped 1. Direct questions are unchanged (0.974 both ways); the loss is in paraphrased (0.933 to 0.803) and vague, symptom-style questions (0.708 to 0.451). Document recall was not hurt, so the damage is at the section level.
- **What the diagnosis showed.** In at least 7 of the 10 hurt cases the reranker put another section of the *correct document* first. In 3 cases (R01, R02, R07) the expected chunk left the top 5. In 5 cases both compared chunks scored 0.03 or less, so the model had little to separate them.
- **A hypothesis that did not survive.** I expected the cross-encoder to follow shared words while the embedder follows meaning. Measured directly, the reranker's first choice shared more of the question's words in only 3 of 10 hurt cases. That explanation is not supported.
- **Blending did not help.** Mixing both scores (cosine weights 0.25, 0.5, 0.75) gave 0.850, 0.863 and 0.890, all below cosine alone (0.913). The weights were tried on the same questions they were scored on, so even a gain would not have counted.
- **Status.** Likely cause: a domain mismatch (a model trained on web-search passages, short IT procedures as documents). **Not tested.** The reranker stays off; no claim is made that it improves retrieval or reduces hallucinations.

## 5. A similarity threshold cannot tell "answerable" from "not answerable"

- **Calibration** (`scripts/calibrate_retrieval.py`, retrieval only, 44 answerable and 22 no-evidence questions). `RETRIEVAL_MIN_SCORE = 0.25` is the highest cut-off that loses no answerable question (0.30 loses one), yet it stops only 1 of 22 questions that have no evidence. Relevant chunks have a median best score of 0.617 and no-evidence questions reach 0.735.
- **Meaning.** Cosine similarity measures how close the *topic* is, not whether the text answers the question. That is why a separate evidence grader decides sufficiency, and why the grader is evaluated.
- **Top-k.** With `top_k = 5`, 2 of 9 multi-document questions miss one expected document (recall 0.889). `k = 8` gives 1.000 but precision falls from 0.291 to 0.199 and every prompt grows. The default stays 5 until answer quality at a larger k is measured.

## 6. A metadata edit that was silently skipped

- **Symptom.** Marking the 2022 password policy `Status: superseded` did not change the index: ingestion reported the file "unchanged".
- **Cause.** The content hash that decides "unchanged" ignored the status line.
- **Fix.** For superseded documents the status is part of the hash (current documents' hashes did not change), and an unchanged file whose stored record disagrees with its header is repaired without re-indexing. A SQLite migration (`schema v2`) adds the columns to existing databases.
- **Lesson.** If a new metadata field changes behaviour, it must be part of the change check.

## 7. A guided live demo found four defects that 396 tests had not

Running every step of `docs/DEMO.md` with real services on 2026-11-21 found:

1. **A typed password was stored in the audit trace.** The question and the final answer were redacted, but the nested `trace` JSON was written raw. Fixed by redacting every string in the trace, grading and validation results (verified: the string appears in no database column on new rows).
2. **Long URLs in stored answers were mangled.** The "any 40-character token" rule, meant for typed text, rewrote URLs and chunk ids to `[REDACTED]` in 13 of 32 external rows. Generated text now gets credential-only redaction.
3. **A correct answer was refused (false abstention).** For "What is the minimum password length?" the model-based validator rejected the right answer twice because it cited both the current policy and the superseded 2022 one. The conflict between a current and a superseded document is now explained by a code-built label, so the model is not asked to judge it. Verified live.
4. **A community Q&A thread was accepted as "official" evidence.** The external grader gave a user-written Microsoft Q&A page credibility 1.0 and an answer's claim traced to it. Pages on community URL paths are now rejected in code. The live re-run did not exercise the override (the model scored those pages 0.3 itself); the 1.0 case is covered by a workflow test.

Also fixed: abstentions now say *why* web search was not used and ask the user to change a password typed into the chat (abstentions only).

- **Lesson.** Tests check what someone thought of. Running the whole system against realistic questions found four things nobody had thought of. None of the evaluation metrics measure the quality of an accepted source, which is why defect 4 was invisible to them.

---

## What is still not proven

- **Agentic beats plain RAG: unproven.** The recorded comparison (24 questions per system, run `20261119T120610Z-f01dbc`) shows similar quality at roughly five times the tokens. Below 30 questions per system the dashboard declines to name a winner, and that run predates later prompt, validator and grader changes. A re-run and an ablation (what each node buys) are planned.
- **No result is on unseen questions yet.** The rules were tuned while looking at failures on these same 93 questions. A frozen dev/test split now separates tuning from reporting ([EVAL_SPLIT.md](EVAL_SPLIT.md)), but the test split was visible during development, so it is less contaminated rather than clean. A block of fresh questions written after the freeze is still missing.
- **The corpus is tiny and synthetic.** With 11 short documents the embedder alone finds the right document almost every time, so retrieval metrics saturate.
- **Judge and run noise.** The faithfulness judge is the same model family as the generator and has not been checked against human labels. The same Teams-outage question produced an answer in one run and an abstention an hour later because the web results differed; a single live run is an observation, not a result.
- **A typed password still reaches the LLM provider.** Web search is blocked and the audit record is redacted, but the text is not removed before the LLM call.
- **Authentication is a local demo** (headers), not identity management.
