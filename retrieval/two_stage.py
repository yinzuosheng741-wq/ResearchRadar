"""Paper-first retrieval using the existing keyword, dense, and hybrid indexes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from time import perf_counter

from agent.execution import ExecutionDenied
from domain.models import EvidenceChunk
from retrieval.hybrid import EvidenceRetriever, HybridRetriever, search_branches
from retrieval.query_expansion import plan_query


@dataclass(frozen=True)
class TwoStageTrace:
    query: str
    stage1_keyword_candidates: int = 0
    stage1_vector_candidates: int = 0
    candidate_paper_ids: list[str] = field(default_factory=list)
    stage2_candidates: int = 0
    selected_count: int = 0
    selected_chunk_ids: list[str] = field(default_factory=list)
    selected_paper_ids: list[str] = field(default_factory=list)
    query_variants: list[str] = field(default_factory=list)
    fallback_used: bool = False
    latency_ms: float = 0.0


def _validate_positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name}_must_be_positive")


def _validate_weight(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"{name}_invalid")


def _unique_papers(chunks: list[EvidenceChunk]) -> list[str]:
    return list(dict.fromkeys(item.paper_id for item in chunks))


def _paper_rrf(
    keyword_papers: list[str],
    vector_papers: list[str],
    *,
    keyword_weight: float,
    vector_weight: float,
    rrf_k: int,
) -> list[str]:
    scores: dict[str, float] = {}
    for ranking, weight in (
        (keyword_papers, keyword_weight),
        (vector_papers, vector_weight),
    ):
        for rank, paper_id in enumerate(ranking, start=1):
            scores[paper_id] = scores.get(paper_id, 0.0) + weight / (rrf_k + rank)
    return [
        paper_id
        for paper_id, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ]


class TwoStageRetriever:
    """Recall papers first, then locate evidence within the selected papers."""

    def __init__(
        self,
        keyword: EvidenceRetriever,
        vector: EvidenceRetriever,
        *,
        keyword_weight: float = 1.0,
        vector_weight: float = 1.0,
        rrf_k: int = 60,
        paper_candidate_k: int = 20,
        paper_k: int = 8,
        chunk_candidate_k: int = 20,
        max_chunks_per_paper: int = 4,
    ) -> None:
        _validate_weight("keyword_weight", keyword_weight)
        _validate_weight("vector_weight", vector_weight)
        if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k < 0:
            raise ValueError("rrf_k_invalid")
        for name, value in (
            ("paper_candidate_k", paper_candidate_k),
            ("paper_k", paper_k),
            ("chunk_candidate_k", chunk_candidate_k),
            ("max_chunks_per_paper", max_chunks_per_paper),
        ):
            _validate_positive(name, value)
        self.keyword = keyword
        self.vector = vector
        self.keyword_weight = float(keyword_weight)
        self.vector_weight = float(vector_weight)
        self.rrf_k = rrf_k
        self.paper_candidate_k = paper_candidate_k
        self.paper_k = paper_k
        self.chunk_candidate_k = chunk_candidate_k
        self.max_chunks_per_paper = max_chunks_per_paper
        self._stage2 = HybridRetriever(
            keyword,
            vector,
            keyword_weight=self.keyword_weight,
            vector_weight=self.vector_weight,
            rrf_k=self.rrf_k,
            candidate_k=self.chunk_candidate_k,
            max_chunks_per_paper=self.max_chunks_per_paper,
        )
        self.last_trace = TwoStageTrace(query="")

    def search(
        self,
        query: str,
        *,
        k: int = 8,
        paper_ids: list[str] | None = None,
    ) -> list[EvidenceChunk]:
        started = perf_counter()
        if k <= 0 or paper_ids == []:
            self.last_trace = TwoStageTrace(query=query, latency_ms=(perf_counter() - started) * 1000)
            return []

        keyword, vector, _failures = search_branches(
            self.keyword, self.vector, query, k=self.paper_candidate_k, paper_ids=paper_ids)
        paper_ids_ranked = _paper_rrf(
            _unique_papers(keyword),
            _unique_papers(vector),
            keyword_weight=self.keyword_weight,
            vector_weight=self.vector_weight,
            rrf_k=self.rrf_k,
        )
        candidate_papers = paper_ids_ranked[: self.paper_k]
        if paper_ids is not None:
            allowed = set(paper_ids)
            candidate_papers = [item for item in candidate_papers if item in allowed]

        fallback_used = False
        if not candidate_papers:
            if paper_ids is not None:
                self.last_trace = TwoStageTrace(
                    query=query,
                    stage1_keyword_candidates=len(keyword),
                    stage1_vector_candidates=len(vector),
                    query_variants=list(plan_query(query).queries),
                    latency_ms=(perf_counter() - started) * 1000,
                )
                return []
            fallback_used = True
            results = self._stage2.search(query, k=k)
        else:
            results = self._stage2.search(query, k=k, paper_ids=candidate_papers)

            # Preserve a global evidence branch so the paper shortlist is not a
            # hard recall bottleneck. An explicit caller scope still applies.
            try:
                global_results = HybridRetriever(
                    self.keyword, self.vector, keyword_weight=self.keyword_weight,
                    vector_weight=self.vector_weight, rrf_k=self.rrf_k,
                    candidate_k=self.chunk_candidate_k,
                    max_chunks_per_paper=self.max_chunks_per_paper,
                ).search(query, k=k, paper_ids=paper_ids)
            except ExecutionDenied:
                raise
            except Exception:
                # An optional recall branch must not erase valid scoped hits.
                global_results = []
                fallback_used = True
            merged = []
            seen = set()
            paper_counts: dict[str, int] = {}
            for index in range(max(len(results), len(global_results))):
                for branch in (results, global_results):
                    if index >= len(branch):
                        continue
                    item = branch[index]
                    if item.chunk_id in seen or paper_counts.get(item.paper_id, 0) >= self.max_chunks_per_paper:
                        continue
                    seen.add(item.chunk_id)
                    paper_counts[item.paper_id] = paper_counts.get(item.paper_id, 0) + 1
                    merged.append(item)
            results = merged[:k]

        stage2_trace = self._stage2.last_trace
        self.last_trace = TwoStageTrace(
            query=query,
            stage1_keyword_candidates=len(keyword),
            stage1_vector_candidates=len(vector),
            candidate_paper_ids=list(candidate_papers),
            stage2_candidates=stage2_trace.fused_candidates,
            selected_count=len(results),
            selected_chunk_ids=[item.chunk_id for item in results[:16]],
            selected_paper_ids=list(dict.fromkeys(item.paper_id for item in results))[:16],
            query_variants=list(plan_query(query).queries)[:6],
            fallback_used=fallback_used,
            latency_ms=(perf_counter() - started) * 1000,
        )
        return results[:k]
