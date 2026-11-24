# Post-Day-3 experiments: three pre-registered changes (written 2026-11-24, before any code or run)

> **Status: PLAN ONLY. No code has been changed and no run has been made for any experiment below.** Everything here is written
> before the first run and must not be edited after a result is seen; a change of plan is a new, dated section. Source of the
> problems: `docs/FINDINGS.md` finding 11 (runs `fresh-1` and `test-1`, frozen code, tag `eval-freeze-1`).
> The synthetic corpus, the 11 core documents and the model (`openai/gpt-oss-120b`) do not change.

## Common design (applies to A, B and C)

1. **One flag per change, default OFF, in `config.py`**, recorded in `settings_snapshot`, so the control and the treatment are the
   same commit with one setting different. The pre-flight check refuses a run whose flags differ from the plan. (Same pattern as
   `RETRIEVAL_MAX_CHUNKS_PER_DOC`.) Flags are named in each section.
2. **A new held-out block, `fresh-2`, written and fingerprinted BEFORE any run and before the flags exist in code.**
   File `data/eval/fresh2_dataset.json` plus `fresh2_dataset.sha256`, guard test like `test_eval_fresh_dataset.py`.
   Rules: written without running the system on the questions or looking at its output for them; disjoint from `dataset.json`,
   `split.json` and `fresh_dataset.json`; the runner needs at least 40 cases covering all eight categories. Planned
   composition (44 cases): direct_kb 6, paraphrased 8, needs_rewriting 6, insufficient_private 4, **current_external 8
   (4 external-only, 4 private-plus-external like E05)**, unanswerable 8, out_of_scope 2, conflicting 2 (conflicting cases follow
   finding 10: reported with and without that category, no conflict claim). At least 12 of the 44 are paraphrase or vague-symptom
   questions whose wording shares few words with the answering sentence (the F11 pattern). The external cases depend on the live
   web, so they are judged on the audit rows as well as the rates.
3. **Same procedure for every arm:** `preflight_eval.py` first, one heavy run per evening, `--no-judge --delay 5`, a distinct run id
   per arm (`f2-control`, `f2-A`, `f2-B`, `f2-C`), never a second run id for the same arm, agentic system only (the baseline does not
   use any of the changed code, so one baseline run `f2-baseline` is enough for the comparison tables). Cost: about 4,200 tokens per
   agentic question, so about 185k per arm; three arms and a control are four evenings. Free-tier limits: about 8k tokens per
   minute, 200k per day.
4. **Metrics fixed in advance** for every arm: overall correct and answerable-correct (keyword scorer, with the known scorer
   artifacts named, labels not edited), hallucinated answers on should-abstain cases, false abstentions, citation correct,
   unnecessary external searches, tokens, and per-case paired differences against `f2-control` using the existing bootstrap
   rule (`better` or `worse` only if the interval excludes zero and at least 10 questions contribute). With 44 questions most
   verdicts will read "not distinguishable"; the decision rules below therefore rest on counts and on reading the saved answers,
   and say so.
5. **Guard rail common to all arms:** an arm is rejected if it produces any hallucinated answer on a should-abstain case that the
   control did not, any new unsupported claim found when the answers are read, any unnecessary external search, or any access-level
   leak. One such case is enough, because these are the behaviours the project promises.
6. **Honest limits to repeat in the report:** same author as the system, known failure patterns in mind, 44 questions, live web
   for the external cases, keyword scoring, judge off, one run per arm (no repeat-run noise estimate).

## Experiment A: decide the external-information label in code (flag `EXTERNAL_LABEL_CHECK_IN_CODE`)

**Observed problem.** In `fresh-1` (F26-F28) the LLM validator rejected three correct, cited, labelled external answers with
`external_clearly_separated=false` (`graph/nodes/final_answer_validator.py`, rule at the `used_external` check), so external fallback was 0/3.
Whether an answer starts with the required line is a fact code can decide (CLAUDE.md, "facts that code can decide").

**Change (treatment).** When the flag is on, `used_external` answers are checked in code: pass the separation check if the rendered
answer starts with the verbatim external-information line, the line is not repeated inside private-cited claims, and every
`[n]` marker that points to an external evidence block refers to a URL that was accepted by the external grader. The LLM validator
is still asked about grounding and unsupported claims; its `external_clearly_separated` field is ignored when the flag is on.
Nothing else changes (no prompt edit, no threshold).

**Hypothesis.** The separation false negatives disappear without letting an unlabelled or unsupported external answer through.

**Primary measure.** Among `fresh-2` cases where external evidence was accepted and a draft was produced, the number rejected with a
`not clearly separated` problem: control versus A. **Secondary:** external-case correct (count) and any external answer in the A arm
that lacks the label line or cites a URL that was not accepted.

**Decision rule.** Adopt A if (i) the A arm has 0 separation rejections among drafts that begin with the label line, (ii) the control arm
shows at least 1 (otherwise the problem did not recur and the result is "not testable on this block"), (iii) the guard rails in the
common design hold, and (iv) reading every external answer in the A arm finds no claim that is not in an accepted source. Otherwise: not
adopted, and the reason is recorded. Unit tests (scripted LLM, zero tokens) must show the code check passing a labelled answer and failing an
unlabelled one before any run; unit tests are not evidence of quality.

## Experiment B: keep supported private chunks when the external half is not established (flag `PARTIAL_PRIVATE_EVIDENCE`)

**Observed problem.** `test-1` E05: "What are our company rules for recording Teams meetings, and is Microsoft reporting any Teams outage today?"
The private grader returned `sufficient=false`, `needs_external_supplement=true`, `supporting_refs=[1]` (confidence 0.95). In
`private_evidence_grader._result`, `effective = sufficient and confidence >= threshold and valid_refs`, so `selected_chunk_ids=[]` and the generator
(`answer_generator`) received only the external page (how to view service health, which does not state an outage) and abstained. The baseline answered
the recording half. The graph has no path for "answer the supported private half and say the external half is not established".

**Change (treatment).** When the flag is on and the classifier intent is `both`, the grader has `needs_external_supplement=true`, confidence is at least the
private threshold and `supporting_refs` is non-empty, the supporting chunks are passed to the generator as private evidence even though the full question
is not covered. The generator prompt gets one added instruction for that case only: answer the part the evidence supports, say plainly which part the
available information does not establish, and do not state the missing part. The validator checks the answer as usual. The routing is not changed
(external search still needs the classifier intent, permission and no sensitive content; `external_eligible` is untouched).

**Hypothesis.** On private-plus-external questions the system returns the supported private half instead of abstaining entirely, without
stating anything the evidence does not support.

**Primary measure.** On the 4 private-plus-external `fresh-2` cases, the number whose answer contains the private expected point and no unsupported
claim (read by hand), control versus B. **Secondary:** answerable-correct overall; any change on the other 40 cases.

**Decision rule.** Adopt B if (i) it recovers the private point on at least 2 of the 4 mixed cases that the control abstained on or got wrong, (ii) the
other 40 cases show no new incorrect answer, no new hallucination and no new external search (a changed answer on a private-only case is a reason to read it and
decide, not an automatic pass), (iii) the common guard rails hold. With 4 mixed cases this is a demonstration, not a rate: report it as such. If the control
already answers them, the result is "not testable on this block".

## Experiment C: paraphrase retrieval (flag `RETRIEVAL_QUERY_EXPANSION`)

**Observed problem.** `fresh-1` F11 ("mine crypto on idle laptops") did not retrieve `Software_Installation_Procedure.txt` in the top 5 (best 0.450); hybrid
BM25 already lost (finding 8) and the reranker hurt (finding 4), so a lexical or cross-encoder change is not tried again.

**Change (treatment).** Before the FIRST retrieval, one extra Groq call produces up to two short paraphrases of the question that use the likely technical
terms ("cryptocurrency miners", "prohibited software") and no new facts; retrieval runs for the original and each paraphrase, results are merged by best
cosine score per chunk, access filtering still runs first, and the original question alone is what the grader and generator see. Cost: about 300 extra
tokens per question. Flag: `RETRIEVAL_QUERY_EXPANSION` (int 0-2, default 0 = off).

**Tuned on dev only.** The rule and the paraphrase prompt are written here and frozen before the run; the only tuning allowed is the choice 1 or 2
paraphrases, made by a retrieval-only comparison on the 65 dev questions (document recall@1/3/5, MRR, nDCG@5, as in `docs/HYBRID_RETRIEVAL.md`), with
`fresh-1` and `test` not used.

**Primary measure.** On the 12-plus paraphrase and vague-symptom `fresh-2` cases, document recall@5 and MRR, control versus C (retrieval rows
are saved per run). **Secondary:** answerable-correct, tokens, and any new wrong answer.

**Decision rule.** Adopt C if (i) document recall@5 on the paraphrase cases improves by at least 2 cases and no case that the control retrieved is lost
(by document, not by chunk score), (ii) on the other cases the five document metrics are not `worse` under the paired rule, (iii) the common guard rails hold.
If hybrid-style harm appears (a document metric `worse`), C is rejected whatever the paraphrase cases show.

## Order and what is NOT planned

1. Write `fresh-2` and fingerprint it (no code, no run). 2. Implement A, B and C behind flags with unit tests (zero tokens), run the CLI and CI checks
the user runs. 3. `f2-control`, then one arm per evening in the order A, B, C. 4. Record a finding after each arm, whatever the outcome. 5. Only an adopted
change's flag default is switched on, in its own commit, after its result is recorded.

Not planned: any change to retrieval thresholds, the grader or classifier prompts, labels, the frozen split or `fresh-1`; a judge run; a claim that the
agentic system is better than the baseline.
