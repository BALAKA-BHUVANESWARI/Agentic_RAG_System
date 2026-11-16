"""Construct and compile the LangGraph StateGraph."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from copilot.graph import routing
from copilot.graph.deps import GraphDeps
from copilot.graph.nodes.answer_generator import make_answer_generator
from copilot.graph.nodes.audit_logger import make_audit_logger
from copilot.graph.nodes.external_evidence_grader import make_external_evidence_grader
from copilot.graph.nodes.external_search import make_external_search
from copilot.graph.nodes.final_answer_validator import make_final_answer_validator
from copilot.graph.nodes.private_evidence_grader import make_private_evidence_grader
from copilot.graph.nodes.private_retriever import make_private_retriever
from copilot.graph.nodes.query_classifier import make_query_classifier
from copilot.graph.nodes.query_rewriter import make_query_rewriter
from copilot.graph.state import CopilotState


def build_graph(deps: GraphDeps):
    """Wire the nine nodes with conditional edges and return the compiled graph."""
    s = deps.settings
    g = StateGraph(CopilotState)
    g.add_node("query_classifier", make_query_classifier(deps))
    g.add_node("private_retriever", make_private_retriever(deps))
    g.add_node("private_evidence_grader", make_private_evidence_grader(deps))
    g.add_node("query_rewriter", make_query_rewriter(deps))
    g.add_node("external_search", make_external_search(deps))
    g.add_node("external_evidence_grader", make_external_evidence_grader(deps))
    g.add_node("answer_generator", make_answer_generator(deps))
    g.add_node("final_answer_validator", make_final_answer_validator(deps))
    g.add_node("audit_logger", make_audit_logger(deps))

    g.add_edge(START, "query_classifier")
    g.add_conditional_edges(
        "query_classifier", routing.route_after_classifier,
        {"private_retriever": "private_retriever", "answer_generator": "answer_generator"},
    )
    g.add_conditional_edges(
        "private_retriever", routing.route_after_retriever,
        {"private_evidence_grader": "private_evidence_grader", "answer_generator": "answer_generator"},
    )
    g.add_conditional_edges(
        "private_evidence_grader", lambda st: routing.route_after_private_grader(st, s),
        {"answer_generator": "answer_generator", "query_rewriter": "query_rewriter", "external_search": "external_search"},
    )
    g.add_conditional_edges(
        "query_rewriter", lambda st: routing.route_after_rewriter(st, s),
        {"private_retriever": "private_retriever", "external_search": "external_search", "answer_generator": "answer_generator"},
    )
    g.add_edge("external_search", "external_evidence_grader")
    g.add_edge("external_evidence_grader", "answer_generator")
    g.add_edge("answer_generator", "final_answer_validator")
    g.add_conditional_edges(
        "final_answer_validator", routing.route_after_validator,
        {"answer_generator": "answer_generator", "audit_logger": "audit_logger"},
    )
    g.add_edge("audit_logger", END)
    return g.compile()
