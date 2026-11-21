# End-to-end demo (Phase 8c)

A guided walkthrough that proves the whole system works, using the two reference questions (VPN and Teams outage) plus the safety checks. Run it yourself and fill in the **Observed** column. Nothing here is a pre-written result: the answers come from the synthetic demo documents and live services, so wording differs between runs.

> Demo documents are **synthetic, not real company policies**. The employee/admin selector is a **local demo, not authentication**.

**Run of 2026-11-21:** the Observed columns below were filled from one live run on that day (reranker off, chunk cap off, `openai/gpt-oss-120b`, roughly 110,000 Groq tokens in total, including the re-checks). They are single runs on live services, so the wording and, for the Teams question, even the outcome (B3 and B4 come from two runs of the same question) change between runs. Four findings from this demo were fixed the same day: the audit record kept a typed password in its trace; the validator rejected a correct answer that cited a current and a superseded document; the external grader accepted a community Q&A thread; and an abstention did not say why web search had not been used.

## 0. Budget and preconditions

- `.env` has valid `GROQ_API_KEY`, `PINECONE_API_KEY` and `TAVILY_API_KEY`.
- The knowledge base is indexed (`python scripts\ingest_kb.py`, or `docker compose run --rm ingest` for Docker).
- **Groq free tier:** `openai/gpt-oss-120b` allows 200,000 tokens per day and 8,000 per minute. This whole demo costs roughly 15,000 to 25,000 tokens. Do not run it on the same day as a full evaluation run unless the daily budget allows both; a `429 ... tokens per day` error means the budget is used up and the demo must wait.
- Run one thing at a time (shared rate limits).

## 0a. Pre-flight checklist (do this first; most steps cost no tokens)

1. **Confirm the knowledge base is exactly the 11 core documents** (Admin > Documents). Extra or demo documents change the answers and make runs incomparable. After the superseded-status change, run `python scripts\ingest_kb.py` once so `Password_Policy_Legacy_2022.txt` is re-indexed with its `Status:` marker (it becomes version 2, and the list must still show 11 documents).
2. **Free checks** (no Groq tokens): `python scripts\run_eval.py --validate-only`, and `pytest -q`.
3. **Check Groq has budget before the first question.** The daily limit is per organization and model and refills gradually (roughly 8,000 tokens per hour), so a new API key does not help. `python scripts\check_groq.py` makes one small call: if it returns a `429`, wait and try again later instead of starting the demo.
4. **Token expectation.** The earlier estimate for this demo is about 15,000 to 25,000 tokens. For comparison, the recorded evaluation run used about 4,600 tokens per agentic question (109,792 tokens over 24 questions), so 10 questions would be near 46,000 in the worst case. Either way it fits in one day's 200,000, but not alongside an evaluation run.
5. **Order that wastes the least if the budget runs out:** health check (free) -> A (VPN) -> B (Teams) -> S1 and S3 (cheap) -> S2, S4, S5, S6, S7 -> section 5 (audit and feedback, free). If a `429` stops you, note which step you reached and continue on another day.
6. **Do not run anything else that uses Groq** while the demo runs (no evaluation, no second browser tab asking questions).
7. **What to keep for the write-up:** the Request ID of A and B (shown under the trace), the `--verbose` output with any key removed, and the filled-in tables. The README screenshots already exist in `docs/screenshots/`, so retake them only if the UI changed.

## 1. Start the system

**Virtualenv** (two terminals, venv active, from `it-support-copilot/`):

```powershell
python scripts\run_api.py                  # terminal 1, leave running
streamlit run frontend\streamlit_app.py    # terminal 2, opens http://localhost:8501
```

**Docker:**

```powershell
docker compose up                          # UI http://127.0.0.1:8501, API docs http://127.0.0.1:8000/docs
```

In the UI sidebar click **Check backend health**. Expected: `Backend: ok`, with every dependency ticked. If one is crossed, fix it before continuing.

## 2. Scenario A: private evidence is sufficient (VPN)

**Command line**

```powershell
python scripts\ask_agent.py "How do I connect to the company VPN from home?" --verbose
```

**UI:** role `employee`, tick **Allow public web search when needed**, ask the same question.

Check:

| # | What to verify | Observed |
|---|---|---|
| A1 | Path is exactly `query_classifier → private_retriever → private_evidence_grader → answer_generator → final_answer_validator → audit_logger` | Pass (2026-11-21, request `a1ec1cb92e884475`): exactly that path. |
| A2 | No `external_search` node and `tavily_calls = 0` | Pass: no `external_search` node, `tavily_calls=0`. |
| A3 | Answer steps come from the VPN guide; citation names `VPN_Connection_Guide.txt` with a section | Pass: the six steps match section 3 of `VPN_Connection_Guide.txt` one for one; cited `[1] VPN_Connection_Guide.txt - section '3. Connecting from home'`. |
| A4 | No external-information warning in the answer | Pass: no warning. |
| A5 | Trace shows retrieved documents, grading (sufficient), validation passed, total time | Pass: 5 chunks retrieved, grader sufficient (0.98), validator grounded, 4,578 tokens, 21.5 s (the first retrieval includes loading the embedding model). |

## 3. Scenario B: private evidence insufficient, public information needed (Teams outage)

```powershell
python scripts\ask_agent.py "What is the latest Microsoft Teams outage guidance?" --verbose
```

UI: same question, public web search ticked.

Check:

| # | What to verify | Observed |
|---|---|---|
| B1 | Private retrieval and grading run first, then up to two `query_rewriter` retries, each followed by retrieval and grading | Pass (CLI, `7eb3635f363a484f`): retrieval and grading, then two rewrites, each followed by retrieval and grading. |
| B2 | Only after the retries does `external_search` run, followed by `external_evidence_grader` | Pass: `external_search` (2 Tavily calls: official domains first, then the wider web) and `external_evidence_grader` ran only after the second retry. |
| B3 | If an external source is accepted: the answer starts with the exact line `External information: This response uses public vendor information and may not represent your company's internal IT policy.` and cites a URL | Seen in a different run of the same question (UI, `52740df2e78f401f`, 2026-11-21 20:17 IST): the answer started with the exact warning line and cited Microsoft Learn pages. That run also accepted a community Q&A thread, which was fixed afterwards. Re-run after the fix (CLI, `cb213d01ce49402f`, 21:18 IST): 1 of 5 results accepted, the official Microsoft Learn page on service health; the two Microsoft Q&A threads carry `community_page: True` and are rejected (the model had scored them low itself this time, so this run shows the check is wired in; the case where the model scores a forum page 1.0 is covered by a workflow test, not by this run); a status monitor and a 2023 news article were rejected as non-official or stale. Answer cited only the official page; 10,254 tokens, 34 s. |
| B4 | If no external source is accepted: the system abstains, recommends the IT Service Desk, and the trace records why | Seen in the CLI run (2026-11-21 19:28 IST): 0 of 5 results accepted (news recaps and a forum thread); abstained, recommended the IT Service Desk, reason `insufficient_evidence` in the trace and the audit record. |
| B5 | The rewritten queries and the external search query appear in the trace; the external query contains no company names or identifiers | Pass: external query `Microsoft Teams outage status` (UI run: `... and guidance`); the rewritten queries are in the trace; no company names or identifiers. |

Live web results vary between runs, so B3 and B4 are both valid outcomes. What must never happen is an answer with no valid citation, or an unlabelled external answer.

## 4. Safety and edge checks

| # | Question / action | Expected | Observed |
|---|---|---|---|
| S1 | UI with **Allow public web search** unticked, ask the Teams question again | No `external_search` node; abstains and points to the IT Service Desk | Pass on the path (CLI `--no-external`): same retry path as B without `external_search`; abstained. The text now says why (after a fix, CLI `42c2929daf2b44ca`): "Public web search was turned off for this request, so no public source was checked." 5,708 tokens. |
| S2 | `python scripts\ask_agent.py "What is the process to request a second monitor for my desk?"` | Company-specific and undocumented: retries, then abstains with **no** web search (regression check for the routing fix) | Pass: two rewrites, abstained, `tavily_calls=0` (the first rewrite retrieved nothing above the 0.25 score floor). |
| S3 | `python scripts\ask_agent.py "What's a good recipe for lasagna?"` | Out of scope: `query_classifier → answer_generator → final_answer_validator → audit_logger`, no retrieval, no search | Pass: classifier, answer_generator, validator, audit_logger; no retrieval, no search; 674 tokens. |
| S4 | Employee asks `How does the service desk verify a caller?`, then admin asks the same | Employee gets no restricted document (abstains); admin gets an answer citing `Service_Desk_Runbook.txt` | Pass with a caveat: the employee got no restricted chunk (`dropped_unauthorized: 0`, the Pinecone filter) and an abstention (`validation_failed`: the grader judged weak chunks sufficient and the validator caught an unsupported claim); the admin got an answer from `Service_Desk_Runbook.txt` section 2. |
| S5 | Ask `What is the minimum password length?` | States the current rule (14 characters) from `Password_Policy.txt`. If it also cites the 2022 policy, the answer starts with an `Outdated source:` line naming `Password_Policy.txt` as its replacement and the Sources list marks it `(superseded)` | First run FAILED (false abstention: the validator rejected the correct answer twice). Fixed 2026-11-21 (a conflict between one current and one superseded source is explained by the code-built label). Re-run: answered with 14 characters plus the 2022 value, `Outdated source:` line and `(superseded)` in Sources, after one revision. |
| S7 | Ask `What was the minimum password length in the 2022 edition of the password policy?` (needs the re-indexed legacy document, pre-flight step 1) | Answers 8 characters, cites `Password_Policy_Legacy_2022.txt`, and shows the `Outdated source:` line | Pass: 8 characters, cites `Password_Policy_Legacy_2022.txt` section 2, with the `Outdated source:` line and `(superseded)` in Sources. |
| S6 | Ask a question containing a fake secret, for example `my password is hunter2, why does Teams fail?` | Classified as sensitive; no web search; the secret is not stored in the audit record | Partly passed at first: classified sensitive (`password disclosed`) and no web search, but the typed password was stored in the audit `trace` (found and fixed 2026-11-21). Re-run: the audit row shows `[REDACTED]`. With a current-information question (`my password is hunter2, what is the latest Microsoft Teams outage guidance?`, CLI `6f6814ae6c674eff`): intent `both`, sensitive (`includes password`), no `external_search`, `tavily_calls=0`; the abstention now says that web search was not used because the message looks sensitive and asks the user to change the password if it is real. Limit: an answered request that contains a typed secret (the first S6 run) gets no such line, and the typed text still reaches the LLM provider. |

## 5. Audit, feedback and administration

1. In the chat, click a **thumbs-up/down** on an answer. Expected: "feedback recorded".
2. Copy the **Request ID** shown under the trace. Look it up:

```powershell
$h = @{ "X-Demo-Role"="admin"; "X-Demo-User"="root" }
Invoke-RestMethod -Uri "http://127.0.0.1:8000/audit/<request_id>" -Headers $h | ConvertTo-Json -Depth 5
```

   Expected: question, intent, retrieved document and chunk ids, rewritten query, grading, external flag and URLs, final answer, citations, validation result, execution time and the feedback you gave. No API keys.
3. Switch the sidebar role to `admin`, open the **Admin** page:
   - **Documents** tab: 11 documents, including the two restricted ones. The `version status` column shows `Superseded by Password_Policy.txt` for `Password_Policy_Legacy_2022.txt` and `Current` for the rest (if the legacy policy shows `Current`, run `python scripts\ingest_kb.py` once; it repairs the record). Upload a small `.txt`, check its status becomes `indexed`, ask a question it answers, then delete it.
   - **Audit** tab: a lookup form. Paste a Request ID and click **Look up** to see the same record as `GET /audit/<request_id>`; it does not list requests.
   - **Evaluation** tab: past runs, the baseline-vs-agentic dashboard, failed cases.

## 6. Record the results

Fill in the tables above, then add screenshots to `docs/screenshots/` using the names the README expects:

| Screenshot | File |
|---|---|
| Chat with a private-evidence answer and citations (scenario A) | `docs/screenshots/Chat-Private.png` |
| Decision trace for the Teams question (scenario B) | `docs/screenshots/External_Trace_Details.png` |
| External-information warning | `docs/screenshots/External_Warning.png` |
| Admin: document management | `docs/screenshots/admin-documents.png` |
| Admin: evaluation dashboard | `docs/screenshots/evaluation-answer-quality.png` |

Do not include API keys, `.env` contents or other secrets in screenshots or pasted output.

## 7. Pass criteria

The demo passes when A1 to A5 hold, B1 to B5 hold for whichever outcome (answer or abstention) occurred, and S1 to S7 all behave as expected. Any deviation is a finding to report, not something to hide: paste the trace and the request id and the cause can be traced from the audit record.
