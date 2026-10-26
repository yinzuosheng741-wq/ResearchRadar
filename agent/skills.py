"""Local manifests and bounded execution for research workflows."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from inspect import signature

from pydantic import BaseModel, ConfigDict, Field

from domain.models import CitedAnswer, ResearchPlan
from agent.execution import ExecutionDenied, consume, current_context, skill_model_budget


class EvidenceQaInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=4000)
    supplemental_queries: list[str] = Field(default_factory=list, max_length=4)


class ResearchPlanInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=4000)
    supplemental_queries: list[str] = Field(default_factory=list, max_length=4)


@dataclass(frozen=True)
class SkillSpec:
    skill_id: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    fallback_text: str
    max_steps: int = 1
    version: str = "1.0"
    requires_evidence: bool = True
    allowed_data_scope: str = "local_core_rag"
    max_model_calls: int = 1

    def __post_init__(self) -> None:
        if self.max_steps <= 0 or self.max_model_calls <= 0:
            raise ValueError("skill_budget_must_be_positive")
        if not self.version.strip() or not self.allowed_data_scope.strip():
            raise ValueError("skill_manifest_field_required")


@dataclass(frozen=True)
class SkillExecution:
    skill_id: str
    output: BaseModel | None
    message: str
    error_code: str | None = None


class SkillRegistry:
    def __init__(self, specs: list[SkillSpec]) -> None:
        self._specs = {spec.skill_id: spec for spec in specs}
        if len(self._specs) != len(specs):
            raise ValueError("duplicate_research_skill")

    def list_specs(self) -> list[SkillSpec]:
        return list(self._specs.values())

    def get(self, skill_id: str) -> SkillSpec:
        try:
            return self._specs[skill_id]
        except KeyError as exc:
            raise ValueError("unknown_research_skill") from exc

    def validate(self, skill_id: str, payload: object) -> BaseModel:
        return self.get(skill_id).input_model.model_validate(payload)


class SkillExecutor:
    def __init__(
        self,
        registry: SkillRegistry,
        *,
        handlers: Mapping[str, Callable[..., object]],
    ) -> None:
        self.registry = registry
        self.handlers = dict(handlers)

    def execute(self, skill_id: str, payload: object) -> SkillExecution:
        spec = self.registry.get(skill_id)
        try:
            context = current_context()
            if context is not None:
                if skill_id not in context.allowed_skills or spec.allowed_data_scope != context.data_scope:
                    context.record("blocked", "skill_scope_denied")
                    raise ExecutionDenied("skill_scope_denied")
            consume("tool_calls")
            request = self.registry.validate(skill_id, payload)
            handler = self.handlers.get(skill_id)
            if handler is None:
                return self._failure(spec)
            values = request.model_dump()
            argument = values.get("query", values.get("paper_ids"))
            parameters = signature(handler).parameters
            kwargs = {}
            if context is not None and context.paper_ids is not None:
                if "paper_ids" not in parameters:
                    raise ExecutionDenied("handler_scope_unsupported")
                kwargs["paper_ids"] = list(context.paper_ids)
            if values.get("supplemental_queries") and "supplemental_queries" in parameters:
                kwargs["supplemental_queries"] = values["supplemental_queries"]
            with skill_model_budget(spec.max_model_calls):
                output = spec.output_model.model_validate(handler(argument, **kwargs))
            if context is not None:
                context.record("finished", skill_id)
        except ExecutionDenied as exc:
            return SkillExecution(skill_id=skill_id, output=None, message=spec.fallback_text,
                                  error_code=str(exc))
        except Exception:
            return self._failure(spec)
        return SkillExecution(
            skill_id=skill_id,
            output=output,
            message="",
        )

    @staticmethod
    def _failure(spec: SkillSpec) -> SkillExecution:
        return SkillExecution(
            skill_id=spec.skill_id,
            output=None,
            message=spec.fallback_text,
            error_code="research_skill_failed",
        )


DEFAULT_SKILL_REGISTRY = SkillRegistry(
    [
        SkillSpec(
            skill_id="evidence_qa",
            max_model_calls=2,
            description="Answer a research question from validated local evidence.",
            input_model=EvidenceQaInput,
            output_model=CitedAnswer,
            fallback_text="当前本地知识库证据不足，无法给出可靠回答。",
        ),
        SkillSpec(
            skill_id="research_plan",
            description="Build a bounded research plan from local literature evidence.",
            input_model=ResearchPlanInput,
            output_model=ResearchPlan,
            fallback_text="当前本地知识库不足以生成可靠研究路线。",
        ),
    ]
)


__all__ = [
    "DEFAULT_SKILL_REGISTRY",
    "EvidenceQaInput",
    "ResearchPlanInput",
    "SkillExecution",
    "SkillExecutor",
    "SkillRegistry",
    "SkillSpec",
]
