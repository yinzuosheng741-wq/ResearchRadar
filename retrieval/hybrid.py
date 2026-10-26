"""Weighted reciprocal-rank fusion for evidence retrieval."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from time import perf_counter
from typing import Protocol

from domain.models import EvidenceChunk
from retrieval.query_expansion import plan_query


class EvidenceRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        k: int,
        paper_ids: list[str] | None = None,
    ) -> list[EvidenceChunk]: ...


@dataclass(frozen=True)
class RetrievalTrace:
    """Bounded diagnostics for one hybrid retrieval call."""

    query: str
    keyword_candidates: int = 0
    vector_candidates: int = 0
    fused_candidates: int = 0
    selected_count: int = 0
    selected_chunk_ids: list[str] = field(default_factory=list)
    selected_paper_ids: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    query_variants: list[str] = field(default_factory=list)
    fallback_used: bool = False
    retrieval_confidence: float = 0.0
    branch_failures: list[str] = field(default_factory=list)


def search_branches(keyword, vector, query: str, *, k: int, paper_ids=None):
    """A broken provider must not discard the other branch's scoped evidence."""
    results, failures = [], []
    for name, retriever in (("keyword", keyword), ("vector", vector)):
        try:
            chunks = retriever.search(query, k=k, paper_ids=paper_ids)
            results.append([c for c in chunks if paper_ids is None or c.paper_id in paper_ids])
        except Exception:
            # No raw provider exception, query or credentials in diagnostics.
            failures.append(name)
            results.append([])
    if len(failures) == 2:
        raise RuntimeError("all_retrieval_branches_failed")
    return results[0], results[1], failures


def reciprocal_rank_fusion(
    keyword: list[EvidenceChunk],
    vector: list[EvidenceChunk],
    k: int = 60,
    *,
    rrf_k: int | None = None,
    keyword_weight: float = 1.0,
    vector_weight: float = 1.0,
) -> list[EvidenceChunk]:
    """Fuse two rankings without mutating their caller-owned chunks."""
    rrf_k = k if rrf_k is None else rrf_k
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")

    scores: dict[str, float] = {}
    chunks: dict[str, EvidenceChunk] = {}
    for ranking, weight in (
        (keyword, keyword_weight),
        (vector, vector_weight),
    ):
        for rank, item in enumerate(ranking, start=1):
            chunks.setdefault(item.chunk_id, item)
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + weight / (
                rrf_k + rank
            )

    return [
        chunks[chunk_id].model_copy(update={"score": score}, deep=True)
        for chunk_id, score in sorted(
            scores.items(), key=lambda item: (-item[1], item[0])
        )
    ]


def _normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


class HybridRetriever:
    """Fuse keyword and vector candidates, then remove scoped duplicates."""

    def __init__(
        self,
        keyword: EvidenceRetriever,
        vector: EvidenceRetriever,
        *,
        keyword_weight: float = 1.0,
        vector_weight: float = 1.0,
        rrf_k: int = 60,
        candidate_k: int = 20,
        allowed_paper_ids: list[str] | None = None,
        max_chunks_per_paper: int = 4,
    ) -> None:
        self.keyword = keyword
        self.vector = vector
        self.keyword_weight = keyword_weight
        self.vector_weight = vector_weight
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self.allowed_paper_ids = allowed_paper_ids
        if max_chunks_per_paper <= 0:
            raise ValueError("max_chunks_per_paper_must_be_positive")
        self.max_chunks_per_paper = max_chunks_per_paper
        self.last_trace = RetrievalTrace(query="")

    def search(
        self,
        query: str,
        *,
        k: int = 8,
        paper_ids: list[str] | None = None,
    ) -> list[EvidenceChunk]:
        started = perf_counter()
        if k <= 0 or paper_ids == []:
            self.last_trace = RetrievalTrace(query=query, latency_ms=(perf_counter() - started) * 1000)
            return []
        scoped_ids = paper_ids
        if self.allowed_paper_ids is not None:
            allowed = set(self.allowed_paper_ids)
            scoped_ids = [item for item in (paper_ids or self.allowed_paper_ids) if item in allowed]
            if not scoped_ids:
                self.last_trace = RetrievalTrace(query=query, latency_ms=(perf_counter() - started) * 1000)
                return []
        keyword, vector, branch_failures = search_branches(
            self.keyword, self.vector, query, k=self.candidate_k, paper_ids=scoped_ids)
        fused = reciprocal_rank_fusion(
            keyword, vector, rrf_k=self.rrf_k,
            keyword_weight=self.keyword_weight, vector_weight=self.vector_weight,
        )
        fallback_used = False
        # Curated-paper filtering is an optimization, not evidence that the
        # corpus has no answer. Retry once without that auto-generated scope.
        if not fused and paper_ids is None and self.allowed_paper_ids is not None:
            fallback_used = True
            keyword, vector, retry_failures = search_branches(
                self.keyword, self.vector, query, k=self.candidate_k, paper_ids=None)
            branch_failures = list(dict.fromkeys([*branch_failures, *retry_failures]))
            fused = reciprocal_rank_fusion(
                keyword, vector, rrf_k=self.rrf_k,
                keyword_weight=self.keyword_weight, vector_weight=self.vector_weight,
            )

        seen: set[tuple[str, int, str]] = set()
        paper_counts: dict[str, int] = {}
        results: list[EvidenceChunk] = []
        for item in fused:
            if paper_ids is not None and item.paper_id not in paper_ids:
                continue
            identity = (item.paper_id, item.page_number, _normalized_text(item.text))
            if identity in seen:
                continue
            if paper_counts.get(item.paper_id, 0) >= self.max_chunks_per_paper:
                continue
            seen.add(identity)
            paper_counts[item.paper_id] = paper_counts.get(item.paper_id, 0) + 1
            results.append(item)
            if len(results) == k:
                break
        self.last_trace = RetrievalTrace(
            query=query,
            keyword_candidates=len(keyword),
            vector_candidates=len(vector),
            fused_candidates=len(fused),
            selected_count=len(results),
            selected_chunk_ids=[item.chunk_id for item in results],
            selected_paper_ids=list(dict.fromkeys(item.paper_id for item in results)),
            latency_ms=(perf_counter() - started) * 1000,
            query_variants=list(plan_query(query).queries),
            fallback_used=fallback_used,
            branch_failures=branch_failures,
            retrieval_confidence=(
                0.0 if not fused else min(1.0, 0.5 + 0.25 * bool(keyword) + 0.25 * bool(vector))
            ),
        )
        return results
