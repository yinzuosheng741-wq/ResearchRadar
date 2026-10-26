import pytest

from domain.models import EvidenceChunk
from retrieval.hybrid import HybridRetriever


class Failed:
    def search(self, *args, **kwargs):
        raise TimeoutError("private provider details")


class Healthy:
    def search(self, *args, **kwargs):
        return [EvidenceChunk(chunk_id=key, paper_id=key, title=key, page_number=1,
                              text=f"evidence {key}") for key in ("allowed", "outside")]


@pytest.mark.parametrize("vector_fails", [True, False])
def test_one_provider_failure_preserves_other_branch_and_scope(vector_fails):
    retriever = HybridRetriever(Healthy() if vector_fails else Failed(),
                                Failed() if vector_fails else Healthy())
    results = retriever.search("evidence", paper_ids=["allowed"])
    assert [chunk.paper_id for chunk in results] == ["allowed"]
    assert retriever.last_trace.branch_failures == ["vector" if vector_fails else "keyword"]


def test_both_providers_fail_without_exposing_provider_details():
    with pytest.raises(RuntimeError, match="^all_retrieval_branches_failed$"):
        HybridRetriever(Failed(), Failed()).search("query")
