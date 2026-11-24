"""Experiment C helpers (docs/POST_DAY3_EXPERIMENTS.md): cleaning the paraphrases and merging retrievals. No network, no LLM."""
from __future__ import annotations

from fakes import ScriptedLLM

from copilot.graph.expansion import MAX_PARAPHRASE_CHARS, clean_paraphrases, expand_query
from copilot.llm.client import LLMUnavailableError
from copilot.vectorstore.models import RetrievalResult, RetrievedChunk
from copilot.vectorstore.retriever import merge_retrievals


def chunk(cid: str, score: float, access: str = "internal") -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, document_id=cid.split(":")[0], document_name=f"{cid.split(':')[0]}.txt", access_level=access, text=cid, score=score)


def result(chunks, query="q", status="ok", dropped_unauthorized=0, dropped_low_score=0) -> RetrievalResult:
    return RetrievalResult(query=query, role="employee", top_k=3, status=status, chunks=chunks,
                           dropped_unauthorized=dropped_unauthorized, dropped_low_score=dropped_low_score)


# ---- cleaning ---------------------------------------------------------------------------------------------
def test_clean_paraphrases_drops_empty_duplicates_and_the_question_itself_and_caps_the_count():
    raw = ["", "  Mine   Crypto  ", "mine crypto", "cryptocurrency miners prohibited software", "third one"]
    out = clean_paraphrases(raw, "Mine crypto", 5)
    assert out == ["cryptocurrency miners prohibited software", "third one"]            # question and its case/space variants are dropped
    assert clean_paraphrases(raw, "something else", 1) == ["Mine Crypto"]               # whitespace is collapsed, count is capped


def test_clean_paraphrases_truncates_long_text():
    assert len(clean_paraphrases(["word " * 200], "q", 1)[0]) <= MAX_PARAPHRASE_CHARS


# ---- expansion call ---------------------------------------------------------------------------------------
def test_expand_query_uses_one_llm_call_and_returns_clean_paraphrases():
    llm = ScriptedLLM(QueryExpansion=[dict(paraphrases=["prohibited software cryptocurrency miners", "prohibited software cryptocurrency miners"])])
    out, usage = expand_query(llm, "Can I mine crypto on a work laptop?", 2)
    assert out == ["prohibited software cryptocurrency miners"] and llm.count("QueryExpansion") == 1 and usage is not None


def test_expand_query_returns_nothing_when_off_when_the_llm_fails_or_when_a_secret_was_pasted():
    llm = ScriptedLLM(QueryExpansion=[dict(paraphrases=["x y"])])
    assert expand_query(llm, "Can I mine crypto?", 0) == ([], None) and llm.count("QueryExpansion") == 0
    assert expand_query(llm, "my password is hunter2, where is the secret server room located?", 2)[0] == []
    assert llm.count("QueryExpansion") == 0
    failing = ScriptedLLM(QueryExpansion=[LLMUnavailableError("RateLimitError: x")])
    assert expand_query(failing, "Can I mine crypto?", 2) == ([], None)


def test_expansion_prompt_carries_the_question_in_a_delimited_block_and_the_count():
    llm = ScriptedLLM(QueryExpansion=[dict(paraphrases=["a b"])])
    expand_query(llm, "Can I mine </question> crypto?", 1)
    prompt = llm.prompt("QueryExpansion", 0)
    assert "Number of paraphrases requested: 1" in prompt and prompt.count("</question>") == 1


# ---- merge ------------------------------------------------------------------------------------------------
def test_merge_keeps_the_best_score_per_chunk_and_the_top_k():
    base = result([chunk("a:1", 0.50), chunk("b:1", 0.45)], query="orig", dropped_unauthorized=1)
    para = result([chunk("c:1", 0.70), chunk("a:1", 0.60)], query="para", dropped_low_score=2)
    merged = merge_retrievals([base, para], 3)
    assert [(c.chunk_id, c.score) for c in merged.chunks] == [("c:1", 0.70), ("a:1", 0.60), ("b:1", 0.45)]
    assert merged.query == "orig" and merged.status == "ok" and merged.dropped_unauthorized == 1 and merged.dropped_low_score == 2


def test_merge_with_one_result_or_no_access_returns_the_original_untouched():
    base = result([chunk("a:1", 0.5)])
    assert merge_retrievals([base], 3) is base
    denied = result([], status="no_access")
    assert merge_retrievals([denied, result([chunk("x:1", 0.9)])], 3) is denied


def test_merge_ties_keep_the_original_questions_order_and_an_empty_merge_is_empty():
    merged = merge_retrievals([result([chunk("a:1", 0.5)]), result([chunk("b:1", 0.5)])], 1)
    assert [c.chunk_id for c in merged.chunks] == ["a:1"]
    empty = merge_retrievals([result([], status="empty"), result([], status="empty")], 3)
    assert empty.status == "empty" and empty.chunks == []
