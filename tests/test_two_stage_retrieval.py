import pytest

from domain.models import EvidenceChunk
from retrieval.two_stage import TwoStageRetriever, TwoStageTrace


def chunk(chunk_id: str, paper_id: str | None = None) -> EvidenceChunk:
    paper_id = paper_id or chunk_id.split(":", 1)[0]
    return EvidenceChunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        title=f"Title {paper_id}",
        page_number=1,
        text=f"evidence {chunk_id}",
    )


class RecordingRetriever:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def search(self, query, *, k, paper_ids=None):
        self.calls.append((query, k, paper_ids))
        filtered = [
            item for item in self.results
            if paper_ids is None or item.paper_id in paper_ids
        ]
        return filtered[:k]


def test_optional_global_branch_failure_keeps_shortlist_evidence():
    class FailsAfterShortlist(RecordingRetriever):
        def search(self, query, *, k, paper_ids=None):
            if len(self.calls) >= 2:
                raise TimeoutError()
            return super().search(query, k=k, paper_ids=paper_ids)

    retriever = TwoStageRetriever(FailsAfterShortlist([chunk("p1:c1")]),
                                  FailsAfterShortlist([chunk("p1:c2")]))
    assert retriever.search("query")
    assert retriever.last_trace.fallback_used


def test_stage_one_collapses_chunk_rankings_to_unique_papers_before_rrf():
    keyword = RecordingRetriever([chunk("p1:c1"), chunk("p1:c2"), chunk("p2:c1")])
    vector = RecordingRetriever([chunk("p2:c2"), chunk("p3:c1")])
    retriever = TwoStageRetriever(
        keyword,
        vector,
        keyword_weight=1.0,
        vector_weight=1.0,
        paper_candidate_k=10,
        paper_k=2,
        chunk_candidate_k=8,
        max_chunks_per_paper=2,
    )

    retriever.search("water quality", k=5)

    assert retriever.last_trace.candidate_paper_ids == ["p2", "p1"]
    assert keyword.calls[0] == ("water quality", 10, None)
    assert vector.calls[0] == ("water quality", 10, None)


def test_two_stage_respects_scope_bounds_and_records_trace():
    keyword = RecordingRetriever([chunk("p1:c1"), chunk("p2:c1")])
    vector = RecordingRetriever([chunk("p2:c2"), chunk("p1:c2")])
    retriever = TwoStageRetriever(keyword, vector, paper_k=2, max_chunks_per_paper=1)

    results = retriever.search("q", k=5, paper_ids=["p2"])

    assert results
    assert all(item.paper_id == "p2" for item in results)
    assert len(results) <= 5
    assert isinstance(retriever.last_trace, TwoStageTrace)
    assert retriever.last_trace.selected_paper_ids == ["p2"]
    assert retriever.last_trace.stage1_keyword_candidates == 1


def test_two_stage_empty_scope_never_falls_back():
    keyword = RecordingRetriever([chunk("p1:c1")])
    vector = RecordingRetriever([chunk("p1:c2")])
    retriever = TwoStageRetriever(keyword, vector)

    assert retriever.search("q", k=3, paper_ids=[]) == []
    assert keyword.calls == []
    assert vector.calls == []


def test_two_stage_falls_back_to_unscoped_hybrid_when_stage_one_is_empty():
    class EmptyStageOne(RecordingRetriever):
        pass

    keyword = EmptyStageOne([])
    vector = EmptyStageOne([])
    retriever = TwoStageRetriever(keyword, vector)

    assert retriever.search("q", k=3) == []
    assert keyword.calls == [("q", 20, None), ("q", 20, None)]
    assert vector.calls == [("q", 20, None), ("q", 20, None)]
    assert retriever.last_trace.fallback_used is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"paper_candidate_k": 0},
        {"paper_k": 0},
        {"chunk_candidate_k": 0},
        {"max_chunks_per_paper": 0},
        {"keyword_weight": -1.0},
    ],
)
def test_two_stage_rejects_invalid_limits_and_weights(kwargs):
    with pytest.raises(ValueError):
        TwoStageRetriever(RecordingRetriever([]), RecordingRetriever([]), **kwargs)
