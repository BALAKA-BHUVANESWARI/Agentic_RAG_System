"""BM25 and Reciprocal Rank Fusion (pure Python, no network, no model)."""
from __future__ import annotations

import pytest

from copilot.vectorstore.lexical import STOPWORDS, BM25Index, reciprocal_rank_fusion, stem, tokenize


# ------------------------------------------------------------------------------ tokenising
@pytest.mark.parametrize("word,expected", [
    ("passwords", "password"), ("policies", "policy"), ("codes", "code"), ("access", "access"), ("status", "status"), ("bus", "bus"),
    ("vpn", "vpn"), ("2fa", "2fa"), ("laptops", "laptop"), ("trees", "tree"), ("goes", "goe"), ("series", "sery"),
])
def test_stem_folds_plurals_and_nothing_else(word, expected):
    # Harman S-stemmer: crude by design ("goes" -> "goe", "series" -> "sery"); it only has to fold query and document the same way.
    assert stem(word) == expected


def test_tokenize_lowercases_drops_stop_words_and_keeps_digits():
    assert tokenize("How do I connect to the company VPN from home?") == ["connect", "company", "vpn", "home"]
    assert tokenize("Reset 2FA codes on IPv4, 8 chars") == ["reset", "2fa", "code", "ipv4", "8", "char"]
    assert tokenize("") == [] and tokenize("the of and") == []
    assert "the" in STOPWORDS and "vpn" not in STOPWORDS


# ------------------------------------------------------------------------------ BM25
DOCS = {
    "vpn": "Connect to the corporate VPN client from home using the SecureLink client",
    "pw": "Password policy: passwords must be 14 characters. Change your password every year",
    "wifi": "Connect to the guest WiFi network in the office",
    "long": "Connect " + " ".join(["filler"] * 200) + " vpn",
}


def test_the_document_with_the_rare_matching_term_ranks_first():
    ix = BM25Index(DOCS)
    top = ix.search("how do I connect to the vpn", 4)
    assert [d for d, _ in top][0] == "vpn" and "pw" not in [d for d, _ in top]
    assert [s for _, s in top] == sorted((s for _, s in top), reverse=True) and top[-1][1] > 0
    assert dict(top)["vpn"] > dict(top)["long"]                                              # same terms, but the long document is diluted
    assert ix.search("password length", 3)[0][0] == "pw"


def test_plurals_match_and_documents_without_a_query_term_are_not_returned():
    ix = BM25Index(DOCS)
    assert [d for d, _ in ix.search("passwords", 5)] == ["pw"]
    assert ix.search("kubernetes", 5) == [] and ix.search("the of and", 5) == [] and ix.search("", 5) == []


def test_a_term_in_fewer_documents_weighs_more():
    ix = BM25Index({"a": "alpha common", "b": "common", "c": "common", "d": "common"})
    assert ix.score("alpha", "a") > ix.score("common", "a") > 0


def test_longer_documents_are_normalised_down():
    ix = BM25Index({"short": "vpn guide", "long": "vpn guide " + " ".join(["filler"] * 100)})
    assert ix.score("vpn", "short") > ix.score("vpn", "long")
    flat = BM25Index({"short": "vpn guide", "long": "vpn guide " + " ".join(["filler"] * 100)}, b=0.0)
    assert flat.score("vpn", "short") == pytest.approx(flat.score("vpn", "long"))             # b=0 switches length normalisation off


def test_repeated_query_terms_do_not_change_the_score():
    ix = BM25Index(DOCS)
    assert ix.score("vpn", "vpn") == ix.score("vpn vpn vpn", "vpn")


def test_the_access_filter_is_applied_before_ranking_and_top_k_counts_allowed_documents_only():
    ix = BM25Index(DOCS)
    assert [d for d, _ in ix.search("connect", 5, allowed=lambda d: d != "vpn")][0] != "vpn"
    assert ix.search("connect", 1, allowed=lambda d: d == "wifi") == [("wifi", pytest.approx(ix.score("connect", "wifi")))]
    assert ix.search("connect", 5, allowed=lambda d: False) == []
    assert len(ix.search("connect", 1)) == 1


def test_ties_are_broken_by_id_so_results_are_deterministic():
    ix = BM25Index({"b": "same words here", "a": "same words here"})
    assert [d for d, _ in ix.search("same", 5)] == ["a", "b"] and ix.search("same", 5) == ix.search("same", 5)


def test_empty_corpus_and_invalid_arguments():
    assert BM25Index({}).search("vpn", 3) == [] and len(BM25Index({})) == 0
    with pytest.raises(ValueError, match="top_k"):
        BM25Index(DOCS).search("vpn", 0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        BM25Index(DOCS, b=1.5)
    with pytest.raises(ValueError, match="k1"):
        BM25Index(DOCS, k1=-1)


# ------------------------------------------------------------------------------ RRF
def test_rrf_rewards_agreement_between_the_rankings():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "d"]])
    assert [d for d, _ in fused] == ["a", "c", "b", "d"]
    assert dict(fused)["a"] == pytest.approx(1 / 61 + 1 / 62) and dict(fused)["d"] == pytest.approx(1 / 63)


def test_an_id_in_only_one_list_still_appears_and_empty_inputs_are_fine():
    assert [d for d, _ in reciprocal_rank_fusion([["a"], []])] == ["a"] and reciprocal_rank_fusion([]) == [] and reciprocal_rank_fusion([[], []]) == []


def test_weights_and_k_change_the_order_predictably():
    assert reciprocal_rank_fusion([["a", "b"], ["b", "a"]], weights=[2.0, 1.0])[0][0] == "a"
    assert reciprocal_rank_fusion([["a", "b"], ["b", "a"]], weights=[1.0, 2.0])[0][0] == "b"
    big, small = reciprocal_rank_fusion([["a"]], k=1), reciprocal_rank_fusion([["a"]], k=100)
    assert big[0][1] > small[0][1]                                                         # a small k makes the top ranks count more


def test_equal_scores_break_ties_by_best_rank_then_id():
    fused = reciprocal_rank_fusion([["x", "y"], ["z", "w"]])                                # x and z tie at rank 1, y and w tie at rank 2
    assert [d for d, _ in fused] == ["x", "z", "w", "y"]


def test_a_repeated_id_in_one_list_counts_once():
    assert dict(reciprocal_rank_fusion([["a", "a", "b"]]))["a"] == pytest.approx(1 / 61)


def test_rrf_rejects_bad_arguments():
    with pytest.raises(ValueError, match="k must"):
        reciprocal_rank_fusion([["a"]], k=0)
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion([["a"]], weights=[1.0, 2.0])
