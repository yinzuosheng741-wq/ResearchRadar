import pytest

from agent.execution import ExecutionContext, execution_scope
from domain.models import EvidenceChunk
from retrieval.coordinator import retrieve_evidence


def chunk(key, paper="allowed"):
    return EvidenceChunk(chunk_id=key, paper_id=paper, title=key, page_number=1, text=key)


def test_later_subquestion_gets_a_slot_instead_of_first_query_truncation():
    class Retriever:
        def search(self, query, *, k, paper_ids=None):
            return [chunk("new-evidence")] if query == "second aspect" else [chunk(f"first-{i}") for i in range(8)]

    result = retrieve_evidence(Retriever(), "original", k=2, supplemental_queries=["second aspect"])
    assert [c.chunk_id for c in result.chunks] == ["first-0", "new-evidence"]
    assert result.audit.unique_candidates == 9


def test_one_failed_branch_preserves_other_evidence_and_never_logs_raw_query(caplog):
    class Retriever:
        def search(self, query, *, k, paper_ids=None):
            if query == "failed secret":
                raise RuntimeError("provider key secret")
            return [chunk("good")]

    result = retrieve_evidence(Retriever(), "failed secret", supplemental_queries=["working"])
    assert result.chunks[0].chunk_id == "good"
    assert result.audit.failed == 1
    assert "secret" not in caplog.text


def test_scope_is_enforced_even_if_retriever_ignores_it():
    class Retriever:
        calls = 0

        def search(self, query, *, k, paper_ids=None):
            self.calls += 1
            assert paper_ids == ["allowed"]
            return [chunk("private", "other"), chunk("public")]

    retriever = Retriever()
    assert [c.chunk_id for c in retrieve_evidence(retriever, "q", paper_ids=["allowed"]).chunks] == ["public"]
    assert retrieve_evidence(retriever, "q", paper_ids=[]).chunks == []
    assert retriever.calls == 1


def test_search_budget_is_shared_and_preserves_partial_results():
    class Retriever:
        def search(self, query, *, k, paper_ids=None):
            return [chunk(query)]

    context = ExecutionContext(max_search_calls=1)
    with execution_scope(context):
        result = retrieve_evidence(Retriever(), "original", supplemental_queries=["next"])
    assert context.search_calls == 1
    assert [c.chunk_id for c in result.chunks] == ["original"]
    assert context.events[-1]["code"] == "search_calls_exhausted"


def test_all_failed_branches_are_reported_as_failure_not_empty_corpus():
    class Retriever:
        def search(self, *args, **kwargs):
            raise TimeoutError()

    with pytest.raises(RuntimeError, match="all_retrieval_branches_failed"):
        retrieve_evidence(Retriever(), "q")
