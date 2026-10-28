"""Policy stops propagate through actual retrieval composition and optional paths."""

from unittest.mock import Mock

import pytest

from agent.execution import ExecutionContext, ExecutionDenied, execution_scope
from retrieval.context import assemble_context
from retrieval.hybrid import HybridRetriever
from retrieval.two_stage import TwoStageRetriever
from tests.test_execution_outcomes import chunks, execute, workflow


@pytest.mark.parametrize("kind", ["evidence_qa", "research_plan"])
@pytest.mark.parametrize("code", ["request_deadline_exceeded", "search_calls_exhausted", "scope_denied"])
def test_branch_policy_denial_reaches_skill_without_trying_another_provider(kind, code):
    keyword = Mock(search=Mock(side_effect=ExecutionDenied(code)))
    vector = Mock(search=Mock(return_value=chunks()))
    retriever = HybridRetriever(keyword, vector)
    service, method = workflow(kind, retriever=retriever)
    result = execute(kind, service, method, ExecutionContext())
    assert (result.status, result.error_code, result.output) == ("blocked", code, None)
    vector.search.assert_not_called()
    service.model.with_structured_output.assert_not_called()


@pytest.mark.parametrize("failure", [False, True])
def test_late_first_branch_cannot_start_second_branch(failure):
    context = ExecutionContext()

    def search(*args, **kwargs):
        context._started -= 1000
        if failure:
            raise TimeoutError("provider failure")
        return chunks()

    vector = Mock(search=Mock(return_value=chunks()))
    with execution_scope(context), pytest.raises(ExecutionDenied, match="request_deadline_exceeded"):
        HybridRetriever(Mock(search=search), vector).search("query")
    vector.search.assert_not_called()


def test_expired_request_never_starts_either_branch():
    keyword, vector = Mock(), Mock()
    with execution_scope(ExecutionContext(timeout_seconds=0)), pytest.raises(ExecutionDenied):
        HybridRetriever(keyword, vector).search("query")
    keyword.search.assert_not_called()
    vector.search.assert_not_called()


def test_provider_failure_still_preserves_other_branch():
    keyword = Mock(search=Mock(side_effect=TimeoutError("private provider failure")))
    vector = Mock(search=Mock(return_value=chunks()))
    retriever = HybridRetriever(keyword, vector)
    with execution_scope(ExecutionContext()):
        assert [item.chunk_id for item in retriever.search("query")] == ["c1", "c2"]
    assert retriever.last_trace.branch_failures == ["keyword"]


def test_optional_global_recall_branch_cannot_swallow_policy_stop():
    keyword = Mock(search=Mock(side_effect=[chunks(), chunks(), ExecutionDenied("scope_denied")]))
    vector = Mock(search=Mock(return_value=chunks()))
    with execution_scope(ExecutionContext()), pytest.raises(ExecutionDenied, match="scope_denied"):
        TwoStageRetriever(keyword, vector).search("query")
    assert keyword.search.call_count == 3
    assert vector.search.call_count == 2


@pytest.mark.parametrize("mode", ["denied", "late_success", "late_failure"])
def test_optional_neighbor_expansion_cannot_hide_policy_stop(mode):
    context = ExecutionContext()

    def neighbors(*args, **kwargs):
        if mode == "denied":
            raise ExecutionDenied("request_deadline_exceeded")
        context._started -= 1000
        if mode == "late_failure":
            raise TimeoutError("private failure")
        return chunks()

    store = Mock(get_chunks_by_ids=Mock(return_value=chunks()), get_chunks_with_context=neighbors)
    with execution_scope(context), pytest.raises(ExecutionDenied, match="request_deadline_exceeded"):
        assemble_context(store, ["c1", "c2"])


def test_late_canonical_load_never_starts_neighbor_load():
    context = ExecutionContext()

    def load(*args, **kwargs):
        context._started -= 1000
        return chunks()

    store = Mock(get_chunks_by_ids=load)
    with execution_scope(context), pytest.raises(ExecutionDenied, match="request_deadline_exceeded"):
        assemble_context(store, ["c1", "c2"])
    store.get_chunks_with_context.assert_not_called()
