from agent.execution import ExecutionContext, consume, current_context, execution_scope
from agent.skills import DEFAULT_SKILL_REGISTRY, SkillExecutor
from domain.models import CitedAnswer


def executor(calls):
    def answer(query, *, paper_ids=None, supplemental_queries=None):
        consume("model_calls")
        calls.append((query, paper_ids, supplemental_queries))
        return CitedAnswer(answer_markdown="ok", evidence_sufficient=True)

    return SkillExecutor(DEFAULT_SKILL_REGISTRY, handlers={"evidence_qa": answer})


def test_denied_skill_does_not_reach_handler():
    calls = []
    with execution_scope(ExecutionContext(allowed_skills=frozenset())):
        result = executor(calls).execute("evidence_qa", {"query": "q"})
    assert result.error_code == "skill_scope_denied"
    assert calls == []


def test_trusted_paper_scope_and_subqueries_reach_handler_with_shared_budget():
    calls = []
    context = ExecutionContext(paper_ids=("p1",))
    with execution_scope(context):
        result = executor(calls).execute("evidence_qa", {"query": "q", "supplemental_queries": ["part"]})
        blocked = executor(calls).execute("evidence_qa", {"query": "again"})
    assert result.output is not None
    assert calls == [("q", ["p1"], ["part"])]
    assert (context.model_calls, context.tool_calls) == (1, 1)
    assert blocked.error_code == "tool_calls_exhausted"
    assert current_context() is None


def test_scope_cannot_be_supplied_in_untrusted_skill_arguments():
    calls = []
    with execution_scope(ExecutionContext(paper_ids=("p1",))):
        result = executor(calls).execute("evidence_qa", {"query": "q", "paper_ids": ["p2"]})
    assert result.output is None
    assert calls == []
