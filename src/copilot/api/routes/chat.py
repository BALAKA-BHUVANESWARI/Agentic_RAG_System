"""POST /chat: run the agentic workflow."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from copilot.api.schemas import ChatRequest, ChatResponse, CitationOut, TraceEventOut
from copilot.api.security import Principal, get_principal
from copilot.api.services import Services, get_services

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse, summary="Ask the IT Support Copilot")
def chat(body: ChatRequest, request: Request, principal: Principal = Depends(get_principal), svc: Services = Depends(get_services)) -> ChatResponse:
    """Run private retrieval, grading, optional rewrite/external fallback and validation; return answer + trace."""
    if len(body.question) > svc.settings.max_question_chars:
        raise HTTPException(422, f"Question is longer than {svc.settings.max_question_chars} characters.")
    r = svc.agent.answer(
        body.question, principal.role, user_id=principal.user_id, session_id=body.session_id,
        external_search_allowed=body.external_search_allowed, request_id=getattr(request.state, "request_id", None),
    )
    a = r.answer
    return ChatResponse(
        request_id=r.request_id, answer=a.rendered, abstained=a.abstained, abstention_reason=a.abstention_reason,
        used_external=a.used_external, citations=[CitationOut(**c.model_dump()) for c in a.citations],
        path=r.path, trace=[TraceEventOut(**e.model_dump()) for e in r.trace], classification=r.classification,
        private_grade=r.private_grade, retrieval_attempts=r.retrieval_attempts, rewritten_queries=r.rewritten_queries,
        retry_count=r.retry_count, external_search_triggered=r.external_search_triggered, external_query=r.external_query,
        external_results=r.external_results, external_grade=r.external_grade, external_blocked_reason=r.external_blocked_reason,
        validation=r.validation, tokens=r.llm_usage.total_tokens, llm_calls=r.llm_usage.calls, tavily_calls=r.tavily_calls,
        latency_ms=r.latency_ms,
    )
