"""The theme builders: fixed text only, and the route line reflects the nodes that ran."""
from __future__ import annotations

from copilot.ui import theme


def test_route_line_marks_only_nodes_that_ran_and_counts_repeats():
    html = theme.route_html(["query_classifier", "private_retriever", "private_retriever", "answer_generator"])
    assert html.count('class="st ran') == 3 and "×2" in html
    assert 'class="st web ran' not in html and "Public web" in html  # present as a station, but not run


def test_route_line_escapes_the_accessible_label():
    assert "<script>" not in theme.route_html(["<script>"])


def test_verdict_kinds():
    assert "verdict none" in theme.verdict_html(True, False, True) and "checked against" not in theme.verdict_html(True, False, True)
    assert "verdict public" in theme.verdict_html(False, True, None)
    assert "verdict company" in theme.verdict_html(False, False, True) and "checked against sources" in theme.verdict_html(False, False, True)
