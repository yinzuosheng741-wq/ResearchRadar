import pytest
from pydantic import ValidationError

from agent.skills import DEFAULT_SKILL_REGISTRY, SkillExecutor
from domain.models import CitedAnswer


def test_registry_exposes_only_two_local_skills():
    specs = DEFAULT_SKILL_REGISTRY.list_specs()

    assert [spec.skill_id for spec in specs] == [
        "evidence_qa",
        "research_plan",
    ]
    assert all(spec.max_steps == 1 for spec in specs)
    assert all(spec.fallback_text for spec in specs)
    assert all(spec.version == "1.0" for spec in specs)
    assert DEFAULT_SKILL_REGISTRY.get("evidence_qa").requires_evidence is True
    assert DEFAULT_SKILL_REGISTRY.get("evidence_qa").allowed_data_scope == "local_core_rag"


def test_skill_spec_rejects_non_positive_budgets():
    from agent.skills import SkillSpec, EvidenceQaInput

    with pytest.raises(ValueError, match="skill_budget_must_be_positive"):
        SkillSpec(
            skill_id="invalid",
            description="invalid",
            input_model=EvidenceQaInput,
            output_model=CitedAnswer,
            fallback_text="fallback",
            max_steps=0,
        )


def test_unknown_skill_is_rejected_before_workflow_call():
    with pytest.raises(ValueError, match="^unknown_research_skill$"):
        DEFAULT_SKILL_REGISTRY.validate("run_sql", {"query": "select *"})


def test_skill_input_schema_is_validated():
    with pytest.raises(ValidationError):
        DEFAULT_SKILL_REGISTRY.validate("evidence_qa", {"query": ""})


def test_skill_executor_validates_output_and_returns_declared_fallback():
    executor = SkillExecutor(
        DEFAULT_SKILL_REGISTRY,
        handlers={
            "evidence_qa": lambda query: CitedAnswer(
                answer_markdown="supported",
                evidence_sufficient=True,
            ),
            "research_plan": lambda query: (_ for _ in ()).throw(
                RuntimeError("provider secret")
            ),
        },
    )

    success = executor.execute("evidence_qa", {"query": "chlorophyll"})
    failure = executor.execute("research_plan", {"query": "plan"})

    assert success.output.answer_markdown == "supported"
    assert success.error_code is None
    assert failure.output is None
    assert failure.error_code == "research_skill_failed"
    assert failure.message == DEFAULT_SKILL_REGISTRY.get(
        "research_plan"
    ).fallback_text
    assert "secret" not in failure.message
