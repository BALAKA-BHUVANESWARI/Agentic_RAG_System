"""Experiment C (docs/POST_DAY3_EXPERIMENTS.md): query expansion before the FIRST private retrieval, behind RETRIEVAL_QUERY_EXPANSION.

Scripted LLM and a stub retriever, so these tests show the wiring and guard rails; they are NOT evidence that expansion helps retrieval.
"""
from __future__ import annotations

from copilot.vectorstore.models import RetrievalResult, RetrievedChunk
from workflow.test_scenarios import CLS_INTERNAL, GRADE_BAD, GRADE_OK, OK_CHECK, PATH_A, REWRITES, answer_draft

QUESTION = "A teammate suggested using our idle work laptops to mine crypto. Is that okay?"
PARAPHRASE = "prohibited software cryptocurrency miners"
MINERS = "Cryptocurrency miners are prohibited on company laptops."


def chunk(cid: str, name: str, score: float, text: str, access: str = "internal") -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, document_id=name.lower(), document_name=name, access_level=access, text=text, score=score)


class StubRetriever:
    """Returns a fixed chunk list per query (an unknown query gets ``default``) and records every query."""

    def __init__(self, by_query: dict[str, list[RetrievedChunk]], default: list[RetrievedChunk]):
        self.by_query, self.default, self.queries = by_query, default, []

    def retrieve(self, query, role=None, **kwargs):
        self.queries.append(query)
        chunks = self.by_query.get(query, self.default)
        return RetrievalResult(query=query, role="employee", top_k=5, status="ok" if chunks else "empty", chunks=chunks)


def stub() -> StubRetriever:
    original = [chunk("sec:1", "Internal_Security_Policy.txt", 0.45, "Lock your screen whenever you leave your laptop.")]
    found = [chunk("sw:5", "Software_Installation_Procedure.txt", 0.62, MINERS)]
    return StubRetriever({PARAPHRASE: found}, default=original)


def test_flag_off_retrieves_once_with_the_original_question_and_makes_no_extra_llm_call(make_agent):
    retr = stub()
    h = make_agent(retriever=retr, QueryClassification=[CLS_INTERNAL], PrivateGrade=[GRADE_OK],
                   AnswerDraft=[answer_draft("Lock your screen [1].")], GroundingCheck=[OK_CHECK])   # a QueryExpansion call would fail the test
    r = h.agent.answer(QUESTION, "employee")
    assert retr.queries == [QUESTION] and r.path == PATH_A and r.retrieval_attempts[0]["expanded_queries"] == []


def test_flag_on_retrieves_for_the_paraphrase_and_merges_before_grading(make_agent):
    retr = stub()
    h = make_agent(retriever=retr, settings_overrides={"retrieval_query_expansion": 1}, QueryClassification=[CLS_INTERNAL],
                   QueryExpansion=[dict(paraphrases=[PARAPHRASE])], PrivateGrade=[GRADE_OK],
                   AnswerDraft=[answer_draft("Cryptocurrency miners are prohibited [1].")], GroundingCheck=[OK_CHECK])
    r = h.agent.answer(QUESTION, "employee")
    assert retr.queries == [QUESTION, PARAPHRASE]
    assert h.llm.names() == ["QueryClassification", "QueryExpansion", "PrivateGrade", "AnswerDraft", "GroundingCheck"]
    grader_prompt = h.llm.prompt("PrivateGrade", 0)
    assert MINERS in grader_prompt                                           # the chunk the paraphrase found reached the grader
    assert PARAPHRASE not in grader_prompt and QUESTION in grader_prompt     # the grader sees the ORIGINAL question only
    assert PARAPHRASE not in h.llm.prompt("AnswerDraft", 0) and r.path == PATH_A
    assert r.retrieval_attempts[0]["query"] == QUESTION and r.retrieval_attempts[0]["expanded_queries"] == [PARAPHRASE]
    assert r.retrieval_attempts[0]["scores"][0]["document"] == "Software_Installation_Procedure.txt"   # best merged score first
    event = next(e for e in r.trace if e.node == "private_retriever")
    assert event.data["expanded_queries"] == [PARAPHRASE]


def test_flag_on_expansion_runs_once_and_never_on_a_rewrite_retry(make_agent):
    retr = stub()
    h = make_agent(retriever=retr, settings_overrides={"retrieval_query_expansion": 1, "max_rewrite_retries": 1},
                   QueryClassification=[CLS_INTERNAL], QueryExpansion=[dict(paraphrases=[PARAPHRASE])],
                   PrivateGrade=[GRADE_BAD, GRADE_OK], QueryRewrite=[REWRITES[0]],
                   AnswerDraft=[answer_draft("Lock your screen [1].")], GroundingCheck=[OK_CHECK])
    r = h.agent.answer(QUESTION, "employee")
    assert h.llm.count("QueryExpansion") == 1
    assert retr.queries == [QUESTION, PARAPHRASE, REWRITES[0]["rewritten_query"]] and r.retry_count == 1


def test_flag_on_a_failed_expansion_changes_nothing(make_agent):
    from copilot.llm.client import LLMUnavailableError

    retr = stub()
    h = make_agent(retriever=retr, settings_overrides={"retrieval_query_expansion": 2}, QueryClassification=[CLS_INTERNAL],
                   QueryExpansion=[LLMUnavailableError("RateLimitError: x")], PrivateGrade=[GRADE_OK],
                   AnswerDraft=[answer_draft("Lock your screen [1].")], GroundingCheck=[OK_CHECK])
    r = h.agent.answer(QUESTION, "employee")
    assert retr.queries == [QUESTION] and r.path == PATH_A and not r.answer.abstained


def test_flag_on_a_failed_paraphrase_lookup_keeps_the_original_result(make_agent):
    class FlakyRetriever(StubRetriever):
        def retrieve(self, query, role=None, **kwargs):
            if query == PARAPHRASE:
                self.queries.append(query)
                raise ConnectionError("index timeout")
            return super().retrieve(query, role, **kwargs)

    retr = FlakyRetriever({}, default=[chunk("sec:1", "Internal_Security_Policy.txt", 0.45, "Lock your screen whenever you leave.")])
    h = make_agent(retriever=retr, settings_overrides={"retrieval_query_expansion": 1}, QueryClassification=[CLS_INTERNAL],
                   QueryExpansion=[dict(paraphrases=[PARAPHRASE])], PrivateGrade=[GRADE_OK],
                   AnswerDraft=[answer_draft("Lock your screen [1].")], GroundingCheck=[OK_CHECK])
    r = h.agent.answer(QUESTION, "employee")
    assert r.path == PATH_A and not r.answer.abstained and r.answer.abstention_reason is None


def test_paraphrase_retrieval_never_returns_a_restricted_document_to_an_employee(make_agent, kb):
    """The real retriever filters by role for every query, so a paraphrase aimed at a restricted runbook gets nothing from it."""
    aimed = "IT service desk runbook ticket priorities P1 P2 first response targets escalation break-glass administrator recovery"
    h = make_agent(settings_overrides={"retrieval_query_expansion": 2, "retrieval_top_k": 8}, QueryClassification=[CLS_INTERNAL],
                   QueryExpansion=[dict(paraphrases=[aimed])], PrivateGrade=[GRADE_OK],
                   AnswerDraft=[answer_draft("Lock your screen [1].")], GroundingCheck=[OK_CHECK])
    r = h.agent.answer("What are the P1 ticket targets?", "employee")
    docs = next(e for e in r.trace if e.node == "private_retriever").data["documents"]
    assert docs and not any(d.startswith(("Service_Desk_Runbook", "Admin_Account_Recovery")) for d in docs)
