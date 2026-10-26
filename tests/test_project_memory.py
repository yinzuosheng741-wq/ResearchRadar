from datetime import UTC, datetime

import pytest

from agent.research_agent import ResearchAgentService
from domain.models import (
    MemoryFact,
    PaperCandidate,
    ResearchProjectMemory,
    ResearchScope,
)
from storage.database import ResearchDatabase


def _candidate(paper_id="p1"):
    return PaperCandidate(
        source="test",
        source_id=paper_id,
        title="Chlorophyll prediction",
        doi=f"10.1234/{paper_id}",
        year=2025,
    )


def test_memory_round_trips_structured_scope(tmp_path):
    database = ResearchDatabase(tmp_path / "research.db")
    memory = ResearchProjectMemory(
        project_id="water-color-prediction",
        topic="lake chlorophyll prediction",
        prediction_target="chlorophyll-a",
        sensors=["Sentinel-2"],
        study_area="Taihu",
        year_range="2020-2025",
        method_constraints=["machine learning"],
        confirmed_paper_ids=["p1"],
        last_active_skill="research_plan",
    )

    database.save_project_memory(memory)

    loaded = database.get_project_memory("water-color-prediction")
    assert loaded.topic == memory.topic
    assert loaded.sensors == ["Sentinel-2"]
    assert loaded.confirmed_paper_ids == ["p1"]
    assert loaded.updated_at.tzinfo is not None


def test_reconfirming_scope_removes_cleared_fields_and_stale_provenance(tmp_path):
    database = ResearchDatabase(tmp_path / "research.db")
    service = ResearchAgentService(model=None, qa_service=None, plan_service=None, memory_store=database)
    service.confirm_scope(ResearchScope(topic="old", sensor="Sentinel-2"))
    service.confirm_scope(ResearchScope(topic="new"))
    memory = database.get_project_memory()
    assert memory.sensors == []
    assert all(f.field != "sensor" for f in memory.facts)
    assert next(f.value for f in memory.facts if f.field == "topic") == "new"


def test_unknown_confirmed_paper_id_is_rejected(tmp_path):
    database = ResearchDatabase(tmp_path / "research.db")

    with pytest.raises(ValueError, match="^paper_not_found$"):
        database.confirm_paper("water-color-prediction", "missing")


def test_scope_update_is_candidate_until_explicit_confirmation(tmp_path):
    database = ResearchDatabase(tmp_path / "research.db")
    database.upsert_candidate(_candidate("p1"))
    service = ResearchAgentService(
        model=None,
        qa_service=None,
        plan_service=None,
        memory_store=database,
    )
    scope = ResearchScope(
        topic="lake chlorophyll prediction",
        prediction_target="chlorophyll-a",
        sensor="Sentinel-2",
    )

    service.confirm_scope(scope)

    assert database.get_project_memory("water-color-prediction").topic == scope.topic


def test_confirm_scope_records_user_confirmed_memory_facts(tmp_path):
    database = ResearchDatabase(tmp_path / "research.db")
    service = ResearchAgentService(
        model=None,
        qa_service=None,
        plan_service=None,
        memory_store=database,
    )

    service.confirm_scope(
        ResearchScope(
            topic="lake chlorophyll prediction",
            prediction_target="chlorophyll-a",
            sensor="Sentinel-2",
        )
    )

    memory = database.get_project_memory()
    facts = {fact.field: fact for fact in memory.facts}
    assert isinstance(facts["topic"], MemoryFact)
    assert facts["topic"].source == "user_confirmed"
    assert facts["sensor"].confidence == 1.0


def test_agent_context_includes_current_project_fields_only(tmp_path):
    database = ResearchDatabase(tmp_path / "research.db")
    database.save_project_memory(
        ResearchProjectMemory(
            topic="lake chlorophyll prediction",
            prediction_target="chlorophyll-a",
            sensors=["Sentinel-2"],
            study_area="Taihu",
            year_range="2020-2025",
            method_constraints=["machine learning"],
            confirmed_paper_ids=["p1"],
            last_active_skill="research_plan",
            updated_at=datetime.now(UTC),
        )
    )

    captured = []

    class Model:
        def with_structured_output(self, schema):
            class Bound:
                def invoke(self, messages):
                    captured.append(messages[1].content)
                    return {
                        "skill_id": "evidence_qa",
                        "rewritten_query": "chlorophyll-a",
                        "scope_updates": {},
                    }

            return Bound()

    class Qa:
        def answer(self, query):
            return {"answer_markdown": "fallback", "evidence_sufficient": False}

    service = ResearchAgentService(
        model=Model(), qa_service=Qa(), plan_service=Qa(), memory_store=database
    )
    service.chat("What predicts chlorophyll-a?")

    assert '"topic": "lake chlorophyll prediction"' in captured[0]
    assert '"sensors": ["Sentinel-2"]' in captured[0]
    assert "research.db" not in captured[0]
    assert "updated_at" not in captured[0]
