# Fresh-2 held-out questions (G01 to G44)

> **Status: written 2026-11-24, before any experiment code and before any run. Report-only; not to be tuned on.** Fingerprint in
> `data/eval/fresh2_dataset.sha256` (guard test `tests/unit/test_eval_fresh2_dataset.py`). Purpose: the judge for the three
> pre-registered experiments in [POST_DAY3_EXPERIMENTS.md](POST_DAY3_EXPERIMENTS.md). Labels refer to the synthetic demo
> knowledge base only; nothing here is real company policy.

## What is in it

44 cases in `data/eval/fresh2_dataset.json` (same schema as `dataset.json`; ids `G01`-`G44`, disjoint from `dataset.json`,
`split.json` and `fresh_dataset.json`). Run it with `EVAL_DATASET_PATH=data\eval\fresh2_dataset.json` and a non-existent
`EVAL_SPLIT_PATH`, for both the pre-flight check and the run, as for `fresh-1`.

| Category | Cases | Notes |
|---|---:|---|
| direct_kb | 6 | one `admin` case (G06, restricted document) |
| paraphrased | 8 | wording shares few words with the answering sentence |
| needs_rewriting | 6 | vague or symptom-style (G15-G20) |
| insufficient_private | 4 | plausible IT questions the documents do not answer |
| current_external | 8 | 4 external-only (G25-G28), 4 private-plus-external (G29-G32, the E05 pattern with different halves) |
| unanswerable | 8 | undocumented or restricted detail; G37 names a restricted-document keyword as a leak check |
| out_of_scope | 2 | |
| conflicting | 2 | kept so the runner's all-eight-categories rule holds; written after finding 10 and expecting the CURRENT rule only (`expects_conflict_notice` false); reported with and without the category; no conflict-handling claim |

Paraphrase or vague-symptom cases (the F11 pattern, experiment C): G07-G14 and G15-G20, 14 in all.
Private-plus-external cases (experiment B): G29-G32. External-only cases (experiment A's separation check also applies to G29-G32): G25-G28.

## How honest is "fresh-2"?

* **Same author as the system, and not blind to the failures.** It was written by the assistant that helped build and analyse the
  system, **after reading the `fresh-1` and `test-1` answers**. The mixed cases (G29-G32) copy the E05 structure on purpose, the
  paraphrase cases copy the F11 pattern, and the external-only cases are stable vendor facts like F26-F28. That makes it a fair
  test of the three changes but a weaker test of anything else. Nothing in the system's output for these exact questions was seen.
* **Small.** 44 questions; 4 mixed cases and 14 paraphrase cases decide experiments B and C, so those results are demonstrations, not rates.
* **External cases depend on the live web.** Whether G25-G32 are answered depends on the search results of that evening. The
  keyword for G31 ("optimize") and G32 ("fido2") assumes Microsoft's current wording and may fail for wording reasons; read the
  answers. The renderer's keyword scoring has known artifacts (singular/plural, paraphrase); labels are not edited after a run.
* **Several facts repeat earlier questions** (recording retention, battery, USB, password reuse), with different wording; the
  documents hold only about 60 distinct facts.
* **No reviewer other than the author.** A human reading the 44 questions and labels before the first run would improve it; any
  edit before the first run is recorded below, and none is allowed after it.

## Rules

1. Report only: no change to prompts, thresholds, routing or labels because of a failure here (a changed case is burned and recorded in `docs/FINDINGS.md`).
2. Run only through the experiment plan: `f2-control` first, one arm per evening, never a second run id for the same arm.
3. Keep the pre-flight check: reranker off, no chunk cap, the dataset's fingerprint, the 11 core documents in the index.

## Edit log

(none: first version, 2026-11-24)

## The questions

| Id | Category | Expected route | Question |
|---|---|---|---|
| G01 | direct_kb | private_only | What am I not allowed to put inside my password? |
| G02 | direct_kb | private_only | How long until I get a loaner laptop if mine will not start at all? |
| G03 | direct_kb | private_only | What must I include in a ticket when my laptop shows a blue screen? |
| G04 | direct_kb | private_only | What are the data classification levels at the company? |
| G05 | direct_kb | private_only | Where can I find the Reset Teams shortcut? |
| G06 | direct_kb | private_only | Who has to approve the use of a break-glass account? |
| G07 | paraphrased | private_only | How much of my laptop's drive should stay empty to keep it running well? |
| G08 | paraphrased | private_only | Where is the button for flagging a suspicious email? |
| G09 | paraphrased | private_only | May I paste a customer's home address into a Teams chat to save time? |
| G10 | paraphrased | private_only | Can I give myself admin rights on my work laptop so I can install tools? |
| G11 | paraphrased | private_only | Do I need the second sign-in step when I open the intranet? |
| G12 | paraphrased | private_only | Am I supposed to warn my colleagues before I record a meeting? |
| G13 | paraphrased | private_only | If I tell IT that my phone has gone missing, what will they do to it? |
| G14 | paraphrased | private_only | Where do I go on the web to register my phone for the second sign-in step? |
| G15 | needs_rewriting | private_only | My laptop dies after barely any time off the charger. |
| G16 | needs_rewriting | private_only | I am away from the office and the HR website will not load. |
| G17 | needs_rewriting | private_only | Nothing happens when I plug my memory stick into my laptop. |
| G18 | needs_rewriting | private_only | I wanted to replay a meeting from four months ago and it is gone. |
| G19 | needs_rewriting | private_only | I downloaded an installer from the internet and my laptop will not let me run it. |
| G20 | needs_rewriting | private_only | My colleague uses a MacBook. Do our software install rules cover her laptop too? |
| G21 | insufficient_private | abstain | How do I request a new keyboard and mouse? |
| G22 | insufficient_private | abstain | How do I change the display name shown on my Teams profile? |
| G23 | insufficient_private | abstain | What is the process for returning my laptop when I leave the company? |
| G24 | insufficient_private | abstain | How many days of notice does IT need to prepare a laptop for a new starter? |
| G25 | current_external | private_then_external | How much storage does Microsoft say a PC needs to install Windows 11? |
| G26 | current_external | private_then_external | What is the name of the Microsoft site where IT administrators can read Windows release information? |
| G27 | current_external | private_then_external | On which day of the month does Microsoft normally release its regular Windows security updates? |
| G28 | current_external | private_then_external | Which Windows 11 editions does Microsoft say can accept incoming Remote Desktop connections? |
| G29 | current_external | private_plus_external | How long are Teams meeting recordings kept at our company, and where does Microsoft say an administrator can see the current status of its cloud services? |
| G30 | current_external | private_plus_external | How often does our laptop guide say to restart, and which version does Microsoft call the last release of Windows 10? |
| G31 | current_external | private_plus_external | Is split tunneling allowed on our VPN, and what does Microsoft recommend for sending Microsoft 365 traffic through a VPN? |
| G32 | current_external | private_plus_external | What does our MFA guide require of administrators besides the authenticator app, and does Microsoft support passwordless sign-in with security keys? |
| G33 | unanswerable | abstain | What is the name of the IT Director? |
| G34 | unanswerable | abstain | What time does the IT service desk open in the morning? |
| G35 | unanswerable | abstain | Which floor of the office is the server room on? |
| G36 | unanswerable | abstain | What is the phone number of the on-call line for critical incidents? |
| G37 | unanswerable | abstain | How long is the sealed break-glass credential envelope valid? |
| G38 | unanswerable | abstain | What is the budget for next year's laptop refresh? |
| G39 | unanswerable | abstain | Which antivirus product runs on company laptops? |
| G40 | unanswerable | abstain | What happens to my Teams chat history when I leave the company? |
| G41 | out_of_scope | out_of_scope | How tall is Mount Everest? |
| G42 | out_of_scope | out_of_scope | Give me a seven-day workout plan. |
| G43 | conflicting | private_only | Is a password of 10 characters acceptable? |
| G44 | conflicting | private_only | Can I reuse the password I had last month? |
