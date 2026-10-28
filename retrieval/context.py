"""Keep canonical retrieval anchors before adding optional neighboring evidence."""

from dataclasses import dataclass

from agent.execution import ExecutionDenied, check_deadline
from domain.models import EvidenceChunk


@dataclass(frozen=True)
class ContextAudit:
    requested_anchors: int = 0
    canonical_anchors: int = 0
    missing_anchors: int = 0
    neighbor_chunks: int = 0
    neighbor_load_failed: bool = False


def assemble_context(store, chunk_ids: list[str], *, paper_ids=None, max_chunks: int = 24,
                     canonical_chunks: list[EvidenceChunk] | None = None):
    """Canonical anchors win over loader ordering, duplicates and stale payloads.

    Only the requested IDs can become anchors. Neighbors must belong to an
    already verified anchor's paper as well as the caller's explicit scope.
    A broken optional context loader must not discard directly retrieved text.
    canonical_chunks, when supplied, must be the trusted store's just-loaded
    result, never retriever or model payloads; this avoids a second database read.
    """
    if max_chunks <= 0 or paper_ids == []:
        return [], ContextAudit()
    check_deadline()
    requested = list(dict.fromkeys(chunk_ids))[:max_chunks]
    allowed = None if paper_ids is None else set(paper_ids)
    loaded = store.get_chunks_by_ids(requested) if canonical_chunks is None else canonical_chunks
    check_deadline()
    canonical = {
        chunk.chunk_id: chunk
        for chunk in loaded
        if chunk.chunk_id in requested and (allowed is None or chunk.paper_id in allowed)
    }
    anchors = [canonical[key] for key in requested if key in canonical]
    anchor_papers = {chunk.paper_id for chunk in anchors}
    selected: dict[str, EvidenceChunk] = {chunk.chunk_id: chunk for chunk in anchors}
    failed = False
    loader = getattr(store, "get_chunks_with_context", None)
    if callable(loader) and anchors and len(selected) < max_chunks:
        try:
            neighbors = loader([chunk.chunk_id for chunk in anchors], window=1)
            check_deadline()
            for chunk in neighbors:
                if chunk.paper_id not in anchor_papers:
                    continue
                if allowed is not None and chunk.paper_id not in allowed:
                    continue
                selected.setdefault(chunk.chunk_id, chunk)
                if len(selected) >= max_chunks:
                    break
        except ExecutionDenied:
            raise
        except Exception:
            failed = True
        check_deadline()
    return list(selected.values()), ContextAudit(
        requested_anchors=len(requested), canonical_anchors=len(anchors),
        missing_anchors=len(requested) - len(anchors),
        neighbor_chunks=len(selected) - len(anchors), neighbor_load_failed=failed,
    )
