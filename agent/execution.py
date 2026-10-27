"""Request-local capabilities, shared budgets and safe execution events."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import monotonic

logger = logging.getLogger(__name__)


class ExecutionDenied(RuntimeError):
    """A stable policy error containing no provider output or user content."""


@dataclass
class ExecutionContext:
    allowed_skills: frozenset[str] = frozenset({"evidence_qa", "research_plan"})
    data_scope: str = "local_core_rag"
    paper_ids: tuple[str, ...] | None = None
    max_model_calls: int = 4
    max_tool_calls: int = 1
    max_search_calls: int = 6
    timeout_seconds: float = 90.0
    model_calls: int = 0
    tool_calls: int = 0
    search_calls: int = 0
    events: list[dict[str, str | int]] = field(default_factory=list)
    _started: float = field(default_factory=monotonic, repr=False)
    _skill_model_limit: int | None = field(default=None, repr=False)

    def check_deadline(self) -> None:
        if monotonic() - self._started >= self.timeout_seconds:
            self.record("blocked", "request_deadline_exceeded")
            raise ExecutionDenied("request_deadline_exceeded")

    def consume(self, resource: str) -> None:
        self.check_deadline()
        maximum = getattr(self, f"max_{resource}")
        if resource == "model_calls" and self._skill_model_limit is not None:
            maximum = min(maximum, self._skill_model_limit)
        used = getattr(self, resource)
        if used >= maximum:
            self.record("blocked", f"{resource}_exhausted")
            raise ExecutionDenied(f"{resource}_exhausted")
        setattr(self, resource, used + 1)
        self.record("started", resource)

    def record(self, status: str, code: str) -> None:
        if len(self.events) < 64:
            self.events.append({"status": status, "code": code})
        logger.info("execution_event status=%s code=%s", status, code)


_context: ContextVar[ExecutionContext | None] = ContextVar("research_execution", default=None)


def current_context() -> ExecutionContext | None:
    return _context.get()


@contextmanager
def execution_scope(context: ExecutionContext):
    token = _context.set(context)
    try:
        yield context
    finally:
        _context.reset(token)


def consume(resource: str) -> None:
    context = current_context()
    if context is not None:
        context.consume(resource)


@contextmanager
def skill_model_budget(max_calls: int):
    context = current_context()
    if context is None:
        yield
        return
    previous = context._skill_model_limit
    ceiling = context.model_calls + max_calls
    context._skill_model_limit = min(previous, ceiling) if previous is not None else ceiling
    try:
        yield
    finally:
        context._skill_model_limit = previous
