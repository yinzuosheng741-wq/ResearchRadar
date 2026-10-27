from types import SimpleNamespace

import pytest
from agent.execution import ExecutionContext
from domain.models import CitedAnswer, ResearchProjectMemory
from web.app import DefaultUiServices


class Model:
    def __init__(self, skill="general_chat"):
        self.skill = skill

    def with_structured_output(self, schema):
        return SimpleNamespace(invoke=lambda messages: {"skill_id": self.skill, "rewritten_query": "question"})

    def invoke(self, messages):
        return SimpleNamespace(content="Configured model reply")


def services(monkeypatch, model):
    service = DefaultUiServices._cli_services()
    monkeypatch.setenv("OPENAI_API_KEY", "local-test-placeholder")
    monkeypatch.setattr("dotenv.load_dotenv", lambda: False)
    monkeypatch.setattr(service, "_model", lambda: model)
    monkeypatch.setattr(service, "_database", lambda: SimpleNamespace(get_project_memory=lambda: ResearchProjectMemory()))
    return service


def forbidden(*args, **kwargs):
    raise AssertionError("Unused dependency must not initialize")


def test_missing_configuration_stops_before_model_database_or_embedding_initialization(monkeypatch):
    service = services(monkeypatch, None)
    for key in ("OPENAI_API_KEY", "OPENAI_ADMIN_KEY"):
        monkeypatch.setenv(key, "")
    monkeypatch.setattr(service, "_model", forbidden)
    monkeypatch.setattr(service, "_database", forbidden)
    monkeypatch.setattr(service, "_vector_store", forbidden)
    with pytest.raises(ValueError, match="research_model_not_configured"):
        service.research_chat("你好")


def test_general_chat_uses_the_real_composition_without_loading_evidence_dependencies(monkeypatch):
    service = services(monkeypatch, Model())
    monkeypatch.setattr(service, "_vector_store", forbidden)
    monkeypatch.setattr("workflows.qa.CitedQaService", forbidden)
    monkeypatch.setattr("workflows.research_plan.ResearchPlanService", forbidden)
    result = service.research_chat("你好")
    assert result.reply.content == "Configured model reply"
    assert result.reply.tool_name == "general_chat"
    assert not result.reply.citations


def test_selected_evidence_workflow_is_lazy_and_keeps_explicit_empty_scope(monkeypatch):
    service = services(monkeypatch, Model("evidence_qa"))
    calls = []

    def vector(database):
        calls.append("vector")
        return object()

    class Qa:
        def __init__(self, retriever, model, *, chunk_store):
            self.retriever = retriever
            calls.append("qa")

        def answer(self, query, *, paper_ids=None, supplemental_queries=None):
            calls.append(("scope", paper_ids))
            return CitedAnswer(answer_markdown="No evidence", evidence_sufficient=False)

    monkeypatch.setattr(service, "_vector_store", vector)
    monkeypatch.setattr("workflows.qa.CitedQaService", Qa)
    monkeypatch.setattr("workflows.research_plan.ResearchPlanService", forbidden)
    service.research_chat("有哪些论文？", execution_context=ExecutionContext(paper_ids=()))
    assert calls == ["vector", "qa", ("scope", [])]


def test_denied_skill_does_not_construct_the_lazy_workflow(monkeypatch):
    service = services(monkeypatch, Model("evidence_qa"))
    monkeypatch.setattr(service, "_vector_store", forbidden)
    result = service.research_chat("有哪些论文？", execution_context=ExecutionContext(allowed_skills=()))
    assert not result.reply.evidence_sufficient
    assert not result.reply.citations
