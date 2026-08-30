"""Bounded conversational research agent over deterministic evidence workflows."""

from __future__ import annotations

import json
import math
from time import perf_counter
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from agent.skills import DEFAULT_SKILL_REGISTRY, SkillExecutor, SkillRegistry

from domain.models import (
    AnswerCitation,
    AgentDiagnostics,
    CitedAnswer,
    ResearchAgentReply,
    ResearchAgentTurn,
    ResearchConversationMessage,
    ResearchConversationState,
    ResearchPlan,
    ResearchScope,
    MemoryFact,
)
from utils.prompt_loader import load_research_agent_prompt
from retrieval.query_expansion import plan_query


MAX_ROUTER_CALLS = 1
MAX_TOOL_CALLS = 1


class ResearchIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill_id: Literal["evidence_qa", "research_plan", "general_chat"] = Field(
        validation_alias=AliasChoices("skill_id", "action")
    )
    rewritten_query: str = Field(min_length=1, max_length=4000)
    scope_updates: ResearchScope = Field(default_factory=ResearchScope)
    intent: str | None = None
    queries: list[str] = Field(default_factory=list, max_length=6)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    clarification_needed: bool = False


class _AgentState(TypedDict, total=False):
    user_text: str
    conversation: dict[str, Any]
    decision: dict[str, Any]
    reply: dict[str, Any]
    model_calls: int
    tool_calls: int
    route_mode: str
    route_reason: str
    route_ms: float


def _escape_markers(text: str) -> str:
    for marker in ("BEGIN UNTRUSTED CONVERSATION", "END UNTRUSTED CONVERSATION"):
        text = text.replace(marker, "[ESCAPED CONVERSATION MARKER]")
    return text


def _fallback_intent(text: str) -> ResearchIntent:
    planned = plan_query(text)
    plan_markers = ("如何下手", "怎么下手", "从哪里开始", "研究路线", "研究方案", "实验设计", "下一步")
    domain_markers = (
        "水色", "遥感", "水质", "叶绿素", "传感器", "卫星", "论文", "文献",
        "RAG", "Agent", "chlorophyll", "remote sensing", "sensor", "paper",
    )
    if any(marker in text for marker in plan_markers):
        skill_id = "research_plan"
    elif any(marker.lower() in text.lower() for marker in domain_markers):
        skill_id = "evidence_qa"
    else:
        skill_id = "general_chat"
    return ResearchIntent(
        skill_id=skill_id,
        rewritten_query=planned.queries[0] if planned.queries else text.strip(),
        intent=planned.intent,
        queries=list(planned.queries),
        confidence=planned.confidence,
        clarification_needed=planned.clarification_needed,
    )


def _fallback_route_reason(skill_id: str) -> str:
    return {
        "research_plan": "fallback_rule_research_plan",
        "evidence_qa": "fallback_rule_domain_question",
        "general_chat": "fallback_rule_general_chat",
    }.get(skill_id, "fallback_rule_general_chat")


def build_research_agent_graph(
    model,
    qa_service,
    plan_service,
    *,
    skill_registry: SkillRegistry = DEFAULT_SKILL_REGISTRY,
    memory_store=None,
):
    """Build a fixed route -> one tool graph; workflows retain evidence authority."""

    handlers = {}
    if qa_service is not None and hasattr(qa_service, "answer"):
        handlers["evidence_qa"] = qa_service.answer
    if plan_service is not None and hasattr(plan_service, "plan"):
        handlers["research_plan"] = plan_service.plan
    executor = SkillExecutor(skill_registry, handlers=handlers)

    def route(state: _AgentState) -> dict[str, Any]:
        started = perf_counter()
        conversation = ResearchConversationState.model_validate(
            state.get("conversation") or {}
        )
        payload = {
            "context_version": 1,
            "summary": conversation.summary[:1500],
            "scope": conversation.scope.model_dump(mode="json"),
            "recent_messages": [
                {
                    "role": message.role,
                    "content": message.content[:2000],
                    "citations": [
                        {
                            "chunk_id": citation.chunk_id,
                            "paper_id": citation.paper_id,
                        }
                        for citation in message.citations[:8]
                    ],
                }
                for message in conversation.messages[-8:]
            ],
            "current_message": state["user_text"],
            "output_requirements": {
                "evidence_required": True,
                "local_corpus_only": True,
                "return_structured_skill": True,
            },
        }
        if memory_store is not None:
            memory = memory_store.get_project_memory()
            payload["project_memory"] = {
                "topic": memory.topic,
                "prediction_target": memory.prediction_target,
                "sensors": memory.sensors,
                "study_area": memory.study_area,
                "year_range": memory.year_range,
                "method_constraints": memory.method_constraints,
                "facts": [fact.model_dump(mode="json") for fact in memory.facts],
            }
        request = [
            SystemMessage(content=load_research_agent_prompt()),
            HumanMessage(
                content=(
                    "BEGIN UNTRUSTED CONVERSATION\n"
                    + _escape_markers(
                        json.dumps(payload, ensure_ascii=False, sort_keys=True)
                    )
                    + "\nEND UNTRUSTED CONVERSATION"
                )
            ),
        ]
        try:
            raw = model.with_structured_output(ResearchIntent).invoke(request)
            decision = ResearchIntent.model_validate(raw)
            planned = plan_query(state["user_text"])
            updates = {
                "intent": planned.intent,
                "queries": list(planned.queries),
                "confidence": planned.confidence,
                "clarification_needed": planned.clarification_needed,
            }
            if planned.intent == "concept_explanation":
                updates.update({
                    "skill_id": "general_chat",
                    "rewritten_query": planned.normalized_query,
                })
            elif planned.intent == "research_plan":
                updates["skill_id"] = "research_plan"
            decision = decision.model_copy(update=updates)
            route_mode = "model"
            route_reason = "model_structured_route"
        except Exception:
            decision = _fallback_intent(state["user_text"])
            route_mode = "fallback"
            route_reason = _fallback_route_reason(decision.skill_id)
        return {
            "decision": decision.model_dump(mode="json"),
            "model_calls": 1,
            "route_mode": route_mode,
            "route_reason": route_reason,
            "route_ms": _safe_ms((perf_counter() - started) * 1000),
        }

    def use_tool(state: _AgentState) -> dict[str, Any]:
        started = perf_counter()
        decision = ResearchIntent.model_validate(state["decision"])
        if decision.skill_id == "general_chat":
            reply = _general_chat_reply(
                model,
                state["user_text"],
                prefix=(
                    "这是概念解释，未调用本地论文证据，以下内容来自通用知识："
                    if decision.intent == "concept_explanation"
                    else None
                ),
            )
        else:
            payload = {"query": decision.rewritten_query}
            execution = executor.execute(decision.skill_id, payload)
            if execution.output is None:
                reply = ResearchAgentReply(
                    content=execution.message,
                    tool_name=decision.skill_id,
                    evidence_sufficient=False,
                )
            elif isinstance(execution.output, ResearchPlan):
                plan = execution.output
                reply = _plan_reply(plan)
                if not plan.evidence_sufficient:
                    reply = _general_chat_reply(
                        model,
                        (
                            "本地知识库没有找到足够的直接证据。请先明确说明这一点，"
                            "再给出一个通用、可验证的研究设计建议，不要伪造论文引用。\n"
                            f"用户问题：{state['user_text']}"
                        ),
                        prefix="本地知识库未找到足够的直接证据，以下是未引用本地论文的通用研究建议：",
                    )
            else:
                answer = CitedAnswer.model_validate(execution.output)
                reply = _answer_reply(answer)
                if (
                    decision.skill_id == "evidence_qa"
                    and not answer.evidence_sufficient
                ):
                    reply = _general_chat_reply(
                        model,
                        (
                            "本地知识库没有找到足够的直接证据。请先明确说明这一点，"
                            "然后基于通用知识回答下面的问题；不要伪造论文引用。\n"
                            f"用户问题：{state['user_text']}"
                        ),
                        prefix="本地知识库未找到足够的直接证据，以下是未引用本地论文的通用解释：",
                    )
        diagnostics = _build_diagnostics(
            skill_id=decision.skill_id,
            route_mode=state.get("route_mode", "fallback"),
            route_reason=state.get("route_reason", "fallback_rule_general_chat"),
            skill_registry=skill_registry,
            workflow={
                "evidence_qa": qa_service,
                "research_plan": plan_service,
            }.get(decision.skill_id),
            reply=reply,
            route_ms=state.get("route_ms", 0.0),
            tool_ms=(perf_counter() - started) * 1000,
        )
        reply = reply.model_copy(update={"diagnostics": diagnostics})
        return {"reply": reply.model_dump(mode="json"), "tool_calls": 1}

    graph = StateGraph(_AgentState)
    graph.add_node("route", route)
    graph.add_node("tool", use_tool)
    graph.add_edge(START, "route")
    graph.add_edge("route", "tool")
    graph.add_edge("tool", END)
    return graph.compile()


class ResearchAgentService:
    def __init__(
        self,
        *,
        model,
        qa_service,
        plan_service,
        skill_registry: SkillRegistry = DEFAULT_SKILL_REGISTRY,
        memory_store=None,
    ) -> None:
        self.memory_store = memory_store
        self.graph = build_research_agent_graph(
            model,
            qa_service,
            plan_service,
            skill_registry=skill_registry,
            memory_store=memory_store,
        )

    def confirm_scope(self, scope: ResearchScope) -> None:
        if self.memory_store is None:
            raise RuntimeError("research_memory_unavailable")
        memory = self.memory_store.get_project_memory()
        now_facts = {fact.field: fact for fact in memory.facts}
        values = {
            "topic": scope.topic.strip(),
            "prediction_target": scope.prediction_target.strip(),
            "sensor": scope.sensor.strip(),
            "study_area": scope.study_area.strip(),
            "year_range": scope.year_range.strip(),
            "method_constraints": ", ".join(
                item.strip() for item in scope.method_constraints if item.strip()
            ),
        }
        for field, value in values.items():
            if value:
                now_facts[field] = MemoryFact(
                    field=field,
                    value=value,
                    source="user_confirmed",
                    confidence=1.0,
                )
        memory = memory.model_copy(
            update={
                "topic": scope.topic,
                "prediction_target": scope.prediction_target,
                "sensors": [scope.sensor] if scope.sensor else memory.sensors,
                "study_area": scope.study_area,
                "year_range": scope.year_range,
                "method_constraints": scope.method_constraints,
                "facts": list(now_facts.values()),
            }
        )
        self.memory_store.save_project_memory(memory)

    def chat(
        self,
        text: str,
        conversation: ResearchConversationState | dict[str, Any] | None = None,
    ) -> ResearchAgentTurn:
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError("research_agent_message_invalid")
        current = (
            conversation
            if isinstance(conversation, ResearchConversationState)
            else ResearchConversationState.model_validate(conversation or {})
        )
        result = self.graph.invoke(
            {
                "user_text": text.strip(),
                "conversation": current.model_dump(mode="json"),
                "model_calls": 0,
                "tool_calls": 0,
            }
        )
        if result.get("model_calls", 0) > MAX_ROUTER_CALLS or result.get("tool_calls", 0) > MAX_TOOL_CALLS:
            raise RuntimeError("research_agent_budget_exceeded")
        decision = ResearchIntent.model_validate(result["decision"])
        reply = ResearchAgentReply.model_validate(result["reply"])
        scope = _merge_scope(current.scope, decision.scope_updates)
        messages = [
            *current.messages,
            ResearchConversationMessage(role="user", content=text.strip()),
            ResearchConversationMessage(
                role="assistant",
                content=reply.content,
                tool_name=reply.tool_name,
                citations=reply.citations,
                diagnostics=reply.diagnostics,
            ),
        ]
        summary = current.summary
        while len(messages) > 8:
            old = messages.pop(0)
            summary = _append_summary(summary, old)
        updated = ResearchConversationState(
            messages=messages,
            summary=summary[-1500:],
            scope=scope,
        )
        return ResearchAgentTurn(reply=reply, state=updated)


def _answer_reply(answer: CitedAnswer) -> ResearchAgentReply:
    content = answer.answer_markdown
    if not answer.evidence_sufficient:
        content = "当前本地知识库证据不足，无法给出可靠回答。"
    return ResearchAgentReply(
        content=content,
        tool_name="evidence_qa",
        citations=answer.citations,
        evidence_sufficient=answer.evidence_sufficient,
        suggested_search_query=answer.suggested_search_query,
    )


def _general_chat_reply(
    model, user_text: str, *, prefix: str | None = None
) -> ResearchAgentReply:
    """Keep ordinary conversation available without pretending it is local evidence."""
    if hasattr(model, "invoke"):
        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "你是一个简洁、友好的普通 AI 助手。当前对话不属于水色遥感科研知识库问答，"
                        "不要伪造论文、引用或本地检索结果。直接回答用户；如果用户随后询问本项目的"
                        "论文事实、方法或研究路线，再建议切换到证据模式。"
                    )
                ),
                HumanMessage(content=user_text),
            ]
        )
        content = getattr(response, "content", response)
    else:
        content = "当前模型未提供普通对话接口，请配置可用模型后重试。"
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    content = str(content).strip()
    if not content:
        content = "我可以继续普通对话；涉及本项目论文事实时，我会切换到带证据的回答。"
    if prefix:
        content = f"{prefix}\n\n{content}"
    return ResearchAgentReply(
        content=content,
        tool_name="general_chat",
        evidence_sufficient=False,
    )


def _plan_reply(plan: ResearchPlan) -> ResearchAgentReply:
    if not plan.evidence_sufficient:
        return ResearchAgentReply(
            content="当前本地知识库不足以生成可靠研究路线，请先补充相关论文。",
            tool_name="research_plan",
            evidence_sufficient=False,
            suggested_search_query=plan.suggested_search_query,
        )
    lines = ["### 文献证据"]
    lines.extend(f"- {finding.text}" for finding in plan.findings)
    lines.append("\n### 建议起步路径")
    lines.extend(
        f"{index}. 待验证步骤：{step}"
        for index, step in enumerate(plan.suggested_steps, start=1)
    )
    citations: list[AnswerCitation] = []
    seen: set[tuple[str, str]] = set()
    for finding in plan.findings:
        for citation in finding.citations:
            key = (citation.chunk_id, citation.quote)
            if key not in seen:
                seen.add(key)
                citations.append(citation)
    return ResearchAgentReply(
        content="\n".join(lines),
        tool_name="research_plan",
        citations=citations,
        evidence_sufficient=True,
    )


def _merge_scope(current: ResearchScope, updates: ResearchScope) -> ResearchScope:
    merged = current.model_dump()
    for name in ("topic", "prediction_target", "sensor", "study_area", "year_range"):
        value = getattr(updates, name).strip()
        if value:
            merged[name] = value[:200]
    if updates.method_constraints:
        merged["method_constraints"] = list(
            dict.fromkeys(item.strip()[:120] for item in updates.method_constraints if item.strip())
        )[:8]
    return ResearchScope.model_validate(merged)


def _append_summary(summary: str, message: ResearchConversationMessage) -> str:
    content = " ".join(message.content.split())[:500]
    return (summary + "\n" + f"{message.role}: {content}").strip()[-1500:]


def _safe_ms(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(number, 3) if math.isfinite(number) and number >= 0 else 0.0


def _trace_value(trace: object, name: str, default: object = 0) -> object:
    return getattr(trace, name, default) if trace is not None else default


def _build_diagnostics(
    *,
    skill_id: str,
    route_mode: str,
    route_reason: str,
    skill_registry: SkillRegistry,
    workflow: object,
    reply: ResearchAgentReply,
    route_ms: object,
    tool_ms: object,
) -> AgentDiagnostics:
    qa_trace = getattr(workflow, "last_trace", None)
    retriever = getattr(workflow, "retriever", None)
    retrieval_trace = getattr(retriever, "last_trace", None)
    keyword_candidates = int(_trace_value(retrieval_trace, "keyword_candidates", 0) or 0)
    vector_candidates = int(_trace_value(retrieval_trace, "vector_candidates", 0) or 0)
    retrieval_candidates = keyword_candidates + vector_candidates
    if retrieval_candidates == 0:
        retrieval_candidates = int(_trace_value(qa_trace, "retrieved_chunks", 0) or 0)
    evidence_chunks = int(_trace_value(qa_trace, "canonical_chunks", 0) or 0)
    if evidence_chunks == 0 and retrieval_trace is not None:
        evidence_chunks = int(_trace_value(retrieval_trace, "selected_count", 0) or 0)
    model_ms = _safe_ms(_trace_value(qa_trace, "model_ms", 0.0))
    retrieval_ms = _safe_ms(_trace_value(qa_trace, "retrieval_ms", 0.0))
    trace_citation_count = _trace_value(qa_trace, "citation_count", None)
    try:
        citation_count = (
            max(int(trace_citation_count), 0)
            if trace_citation_count is not None
            else len(reply.citations)
        )
    except (TypeError, ValueError):
        citation_count = len(reply.citations)
    total_ms = max(
        _safe_ms(route_ms) + _safe_ms(tool_ms),
        model_ms + retrieval_ms,
    )
    allowed_reasons = {
        "model_structured_route",
        "fallback_rule_research_plan",
        "fallback_rule_domain_question",
        "fallback_rule_general_chat",
    }
    if skill_id == "general_chat":
        skill_version = "not_applicable"
    else:
        try:
            skill_version = skill_registry.get(skill_id).version
        except ValueError:
            skill_version = "unknown"
    return AgentDiagnostics(
        route_mode=route_mode if route_mode in {"model", "fallback"} else "fallback",
        route_reason=(
            route_reason
            if route_reason in allowed_reasons
            else "fallback_rule_general_chat"
        ),
        skill_id=skill_id,
        skill_version=skill_version,
        evidence_sufficient=reply.evidence_sufficient,
        retrieval_candidates=max(retrieval_candidates, 0),
        evidence_chunks=max(evidence_chunks, 0),
        citation_count=citation_count,
        fallback=(
            route_mode != "model"
            or (skill_id != "general_chat" and not reply.evidence_sufficient)
        ),
        retrieval_ms=retrieval_ms,
        model_ms=model_ms,
        total_ms=_safe_ms(total_ms),
    )


__all__ = [
    "MAX_ROUTER_CALLS",
    "MAX_TOOL_CALLS",
    "ResearchAgentService",
    "ResearchIntent",
    "build_research_agent_graph",
]
