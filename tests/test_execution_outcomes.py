"""Exercise policy and failure boundaries through real research workflows."""

from unittest.mock import Mock

import pytest
from agent.evidence_review import EvidenceAssessment
from agent.execution import ExecutionContext, ExecutionDenied, execution_scope
from agent.research_agent import ResearchAgentService, ResearchIntent
from agent.skills import DEFAULT_SKILL_REGISTRY, SkillExecutor
from domain.models import CitedAnswer, EvidenceChunk
from retrieval.coordinator import retrieve_evidence
from workflows.qa import CitedQaService
from workflows.research_plan import ResearchPlanService


def chunks():
    return [EvidenceChunk(chunk_id=f"c{i}", paper_id=f"p{i}", title=f"Paper {i}",
                          page_number=i, text=f"Canonical evidence about method {i}.") for i in (1, 2)]


def workflow(kind, *, retriever=None, model=None):
    evidence = chunks()
    retriever = retriever or Mock(search=Mock(return_value=evidence))
    model = model or Mock()
    retriever.last_trace = None
    store = Mock(spec=["get_chunks_by_ids"], get_chunks_by_ids=Mock(return_value=evidence))
    if kind == "evidence_qa":
        return CitedQaService(retriever, model, chunk_store=store), "answer"
    return ResearchPlanService(retriever, model, chunk_store=store), "plan"


def execute(kind, service, method, context, query="Which methods are present?"):
    runner = SkillExecutor(DEFAULT_SKILL_REGISTRY, handlers={kind: getattr(service, method)})
    with execution_scope(context):
        return runner.execute(kind, {"query": query})


@pytest.mark.parametrize("kind", ["evidence_qa", "research_plan"])
@pytest.mark.parametrize("resource", ["search_calls", "model_calls"])
def test_zero_budget_propagates_through_workflow_to_blocked_outcome(kind, resource):
    service, method = workflow(kind)
    context = ExecutionContext(**{f"max_{resource}": 0})
    result = execute(kind, service, method, context)
    assert result.status == "blocked"
    assert result.error_code == f"{resource}_exhausted"
    assert result.output is None
    service.model.with_structured_output.assert_not_called()
    if resource == "search_calls":
        service.retriever.search.assert_not_called()


@pytest.mark.parametrize("kind", ["evidence_qa", "research_plan"])
def test_retrieval_outage_is_failed_not_an_empty_corpus(kind):
    retriever = Mock(search=Mock(side_effect=RuntimeError("private provider key")))
    service, method = workflow(kind, retriever=retriever)
    result = execute(kind, service, method, ExecutionContext())
    assert result.status == "failed" and result.error_code == "retrieval_failed"
    assert result.output is None and "未判定" in result.message
    assert "private" not in result.message
    service.model.with_structured_output.assert_not_called()


@pytest.mark.parametrize("kind", ["evidence_qa", "research_plan"])
def test_model_outage_is_failed_not_insufficient_evidence(kind):
    model = Mock()
    model.with_structured_output.return_value.invoke.side_effect = RuntimeError("private provider key")
    service, method = workflow(kind, model=model)
    result = execute(kind, service, method, ExecutionContext())
    assert result.status == "failed" and result.error_code == "model_failed"
    assert result.output is None and "private" not in result.message


@pytest.mark.parametrize("kind", ["evidence_qa", "research_plan"])
def test_successful_empty_search_remains_an_evidence_outcome(kind):
    retriever = Mock(search=Mock(return_value=[]))
    service, method = workflow(kind, retriever=retriever)
    service.chunk_store.get_chunks_by_ids.return_value = []
    result = execute(kind, service, method, ExecutionContext())
    assert result.status == "insufficient_evidence" and result.error_code is None
    assert result.output is not None and not result.output.evidence_sufficient


def test_reviewer_budget_denial_is_not_swallowed_as_an_optional_failure():
    service, method = workflow("evidence_qa")
    result = execute("evidence_qa", service, method, ExecutionContext(max_model_calls=0), query="Compare methods")
    assert result.status == "blocked" and result.error_code == "model_calls_exhausted"
    service.model.with_structured_output.assert_not_called()


def test_supplemental_search_deadline_cannot_continue_with_initial_evidence():
    model = Mock()
    model.with_structured_output.return_value.invoke.return_value = EvidenceAssessment(supplemental_queries=["missing aspect"])
    retriever = Mock(search=Mock(side_effect=[chunks(), ExecutionDenied("request_deadline_exceeded")]))
    service, method = workflow("evidence_qa", model=model, retriever=retriever)
    result = execute("evidence_qa", service, method, ExecutionContext(), query="Compare methods")
    assert result.status == "blocked" and result.error_code == "request_deadline_exceeded"
    assert model.with_structured_output.call_count == 1  # Reviewer only, no answer writer.


def test_optional_search_budget_can_preserve_existing_evidence_within_model_budget():
    model = Mock()
    evidence = chunks()
    citation = {"chunk_id": "c1", "paper_id": "p1", "title": "Paper 1",
                "page_number": 1, "quote": evidence[0].text}
    model.with_structured_output.return_value.invoke.side_effect = [
        EvidenceAssessment(supplemental_queries=["missing aspect"]),
        CitedAnswer(answer_markdown="Supported excerpt", citations=[citation], evidence_sufficient=True),
    ]
    service, method = workflow("evidence_qa", model=model)
    context = ExecutionContext(max_search_calls=1, max_model_calls=2)
    result = execute("evidence_qa", service, method, context, query="Compare methods")
    assert result.status == "success"
    assert result.output.citations[0].chunk_id == "c1"
    assert (context.search_calls, context.model_calls) == (1, 2)
    assert service.retriever.search.call_count == 1


def test_deadline_after_successful_search_does_not_reuse_partial_results():
    context = ExecutionContext()

    def search(*args, **kwargs):
        context._started -= 1000
        return chunks()

    with execution_scope(context), pytest.raises(ExecutionDenied, match="request_deadline_exceeded"):
        retrieve_evidence(Mock(search=search), "original", supplemental_queries=["next"])


def test_completed_but_late_skill_output_is_rejected():
    context = ExecutionContext()

    def answer(query):
        context._started -= 1000
        return CitedAnswer(answer_markdown="late result", evidence_sufficient=True)

    runner = SkillExecutor(DEFAULT_SKILL_REGISTRY, handlers={"evidence_qa": answer})
    with execution_scope(context):
        result = runner.execute("evidence_qa", {"query": "q"})
    assert result.status == "blocked" and result.error_code == "request_deadline_exceeded"
    assert result.output is None


def test_invalid_input_and_missing_handler_have_distinct_safe_codes():
    runner = SkillExecutor(DEFAULT_SKILL_REGISTRY, handlers={})
    invalid = runner.execute("evidence_qa", {"query": ""})
    unavailable = runner.execute("evidence_qa", {"query": "q"})
    assert (invalid.status, invalid.error_code) == ("invalid_input", "skill_input_invalid")
    assert (unavailable.status, unavailable.error_code) == ("failed", "skill_handler_unavailable")


class AgentModel:
    def __init__(self):
        self.general_calls = 0
        self.route_calls = 0

    def with_structured_output(self, schema):
        assert schema is ResearchIntent
        return self

    def invoke(self, messages):
        if "BEGIN UNTRUSTED CONVERSATION" in str(messages[-1].content):
            self.route_calls += 1
            return {"skill_id": "evidence_qa", "rewritten_query": "Which methods are present?"}
        self.general_calls += 1
        return type("Response", (), {"content": "general answer"})()


@pytest.mark.parametrize("kind", ["blocked", "failed", "insufficient_evidence"])
def test_full_agent_fallback_depends_on_execution_outcome(kind):
    model = AgentModel()
    if kind == "blocked":
        service, _ = workflow("evidence_qa")
        context = ExecutionContext(max_search_calls=0)
    elif kind == "failed":
        service, _ = workflow("evidence_qa", retriever=Mock(search=Mock(side_effect=RuntimeError("private"))))
        context = ExecutionContext()
    else:
        service, _ = workflow("evidence_qa", retriever=Mock(search=Mock(return_value=[])))
        service.chunk_store.get_chunks_by_ids.return_value = []
        context = ExecutionContext()
    turn = ResearchAgentService(model=model, qa_service=service, plan_service=None).chat(
        "论文用了哪些方法？", execution_context=context,
    )
    assert turn.reply.diagnostics.execution_status == kind
    assert model.general_calls == (1 if kind == "insufficient_evidence" else 0)
    assert turn.reply.citations == []
    assert turn.state.messages[-1].diagnostics.execution_status == kind


def test_router_budget_denial_stops_before_any_model_or_workflow_call():
    model = AgentModel()
    qa = Mock()
    turn = ResearchAgentService(model=model, qa_service=qa, plan_service=None).chat(
        "论文用了哪些方法？", execution_context=ExecutionContext(max_model_calls=0),
    )
    assert turn.reply.diagnostics.execution_status == "blocked"
    assert turn.reply.diagnostics.execution_error == "model_calls_exhausted"
    assert model.route_calls == model.general_calls == 0
    qa.answer.assert_not_called()


def test_old_saved_diagnostics_do_not_acquire_a_fictitious_success_status():
    from domain.models import AgentDiagnostics

    old = AgentDiagnostics.model_validate({"route_mode": "fallback", "route_reason": "fallback_rule_domain_question",
                                          "skill_id": "evidence_qa", "evidence_sufficient": False})
    assert old.execution_status == "unknown"


def test_late_general_fallback_does_not_escape_request_deadline():
    context = ExecutionContext()

    class LateModel(AgentModel):
        def invoke(self, messages):
            result = super().invoke(messages)
            if self.general_calls:
                context._started -= 1000
            return result

    model = LateModel()
    service, _ = workflow("evidence_qa", retriever=Mock(search=Mock(return_value=[])))
    service.chunk_store.get_chunks_by_ids.return_value = []
    turn = ResearchAgentService(model=model, qa_service=service, plan_service=None).chat(
        "论文用了哪些方法？", execution_context=context,
    )
    assert model.general_calls == 1
    assert turn.reply.content != "general answer"
    assert turn.reply.diagnostics.execution_status == "blocked"
    assert turn.reply.diagnostics.execution_error == "request_deadline_exceeded"
