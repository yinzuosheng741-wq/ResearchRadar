"""Bounded multi-query retrieval with rank fusion and immutable paper scope."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from time import perf_counter

from domain.models import EvidenceChunk
from agent.execution import ExecutionDenied, consume
from retrieval.query_expansion import plan_query

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchAudit:
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    unique_candidates: int = 0
    selected: int = 0
    branch_counts: tuple[int, ...] = ()
    query_hashes: tuple[str, ...] = ()
    selected_chunk_ids: tuple[str, ...] = ()
    latency_ms: float = 0.0
    # This reports execution coverage, not calibrated relevance confidence.
    scope_applied: bool = False
    budget_exhausted: bool = False
    candidate_chunk_ids: tuple[str, ...] = ()


def merge_audits(first: SearchAudit, second: SearchAudit, selected_ids: list[str]) -> SearchAudit:
    candidates = tuple(dict.fromkeys((*first.candidate_chunk_ids, *second.candidate_chunk_ids)))
    return SearchAudit(
        attempted=first.attempted + second.attempted,
        succeeded=first.succeeded + second.succeeded,
        failed=first.failed + second.failed,
        unique_candidates=len(candidates), candidate_chunk_ids=candidates,
        selected=len(selected_ids), selected_chunk_ids=tuple(selected_ids),
        branch_counts=first.branch_counts + second.branch_counts,
        query_hashes=first.query_hashes + second.query_hashes,
        latency_ms=first.latency_ms + second.latency_ms,
        scope_applied=first.scope_applied or second.scope_applied,
        budget_exhausted=first.budget_exhausted or second.budget_exhausted,
    )


@dataclass
class SearchResult:
    chunks: list[EvidenceChunk] = field(default_factory=list)
    audit: SearchAudit = field(default_factory=SearchAudit)


def retrieve_evidence(
    retriever,
    question: str,
    *,
    k: int = 8,
    paper_ids: list[str] | None = None,
    supplemental_queries: list[str] | None = None,
    max_queries: int = 6,
    candidate_k: int = 24,
) -> SearchResult:
    """Fuse all successful branches before truncation; never broaden explicit scope.

    Searches run sequentially because existing retrievers expose mutable last_trace
    and use a single SQLite connection. Parallelism requires per-worker stores.
    """
    started = perf_counter()
    if k <= 0 or paper_ids == []:
        return SearchResult(audit=SearchAudit(scope_applied=paper_ids is not None))
    if not 1 <= max_queries <= 6 or not 1 <= candidate_k <= 100:
        raise ValueError("retrieval_budget_invalid")
    plan = plan_query(question)
    # Reserve capacity for independently planned subquestions before synonyms.
    extra = [q.strip()[:4000] for q in (supplemental_queries or [])[:4] if isinstance(q, str) and q.strip()]
    queries = list(dict.fromkeys([question, *extra, *plan.queries]))[:max_queries]
    allowed = set(paper_ids) if paper_ids is not None else None
    rankings: list[list[str]] = []
    chunks: dict[str, EvidenceChunk] = {}
    scores: dict[str, float] = {}
    hashes: list[str] = []
    failed = 0
    exhausted = False
    logger.info("retrieval_started branches=%d scoped=%s", len(queries), allowed is not None)
    for query in queries:
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
        try:
            consume("search_calls")
            hashes.append(query_hash)
            found = retriever.search(query, k=max(k, candidate_k), paper_ids=paper_ids)
            ranking: list[str] = []
            seen: set[str] = set()
            for chunk in found:
                if allowed is not None and chunk.paper_id not in allowed:
                    continue
                if chunk.chunk_id in seen:
                    continue
                seen.add(chunk.chunk_id)
                chunks.setdefault(chunk.chunk_id, chunk)
                ranking.append(chunk.chunk_id)
            rankings.append(ranking)
            for rank, chunk_id in enumerate(ranking, 1):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (60 + rank)
            logger.info("retrieval_branch_finished query_hash=%s count=%d", query_hash, len(ranking))
        except ExecutionDenied as exc:
            # Partial search coverage can stop query expansion. A request
            # deadline or any other policy denial must stop the whole task.
            if str(exc) != "search_calls_exhausted":
                raise
            exhausted = True
            if not rankings:
                raise
            break
        except Exception:
            failed += 1
            logger.warning("retrieval_branch_failed query_hash=%s", query_hash)
    if failed == len(queries):
        raise RuntimeError("all_retrieval_branches_failed")
    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    # Give each subquery one representative before filling the remaining budget.
    representatives = list(dict.fromkeys(ranking[0] for ranking in rankings if ranking))
    selected_ids = list(dict.fromkeys([*representatives, *ordered]))[:k]
    selected = [chunks[key].model_copy(update={"score": scores[key]}) for key in selected_ids]
    audit = SearchAudit(
        attempted=len(hashes), succeeded=len(rankings), failed=failed,
        unique_candidates=len(chunks), selected=len(selected),
        branch_counts=tuple(len(ranking) for ranking in rankings),
        query_hashes=tuple(hashes), selected_chunk_ids=tuple(selected_ids),
        latency_ms=round((perf_counter() - started) * 1000, 3),
        scope_applied=allowed is not None,
        budget_exhausted=exhausted,
        candidate_chunk_ids=tuple(chunks),
    )
    logger.info("retrieval_finished candidates=%d selected=%d failed=%d", len(chunks), len(selected), failed)
    return SearchResult(chunks=selected, audit=audit)
