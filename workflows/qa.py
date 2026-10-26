"""Citation-safe, evidence-only question answering."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from time import perf_counter
from typing import Protocol

from langchain_core.messages import HumanMessage, SystemMessage

from domain.models import AnswerCitation, AnswerClaim, CitedAnswer, EvidenceChunk
from agent.execution import consume
from agent.evidence_review import EvidenceAssessment, needs_evidence_review, review_evidence
from retrieval.coordinator import retrieve_evidence, SearchAudit, merge_audits
from retrieval.query_expansion import plan_query
from retrieval.context import ContextAudit, assemble_context
from utils.prompt_loader import load_cited_qa_prompt


class Retriever(Protocol):
    def search(
        self,
        query: str,
        *,
        k: int,
        paper_ids: list[str] | None = None,
    ) -> list[EvidenceChunk]: ...


class ChunkStore(Protocol):
    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[EvidenceChunk]: ...


@dataclass(frozen=True)
class QATrace:
    """Safe runtime diagnostics for one evidence QA request."""

    status: str
    retrieval_ms: float = 0.0
    model_ms: float = 0.0
    retrieved_chunks: int = 0
    canonical_chunks: int = 0
    citation_count: int = 0
    original_query: str = ""
    query_variants: list[str] = field(default_factory=list)
    intent_confidence: float = 0.0
    retrieval_confidence: float = 0.0
    evidence_confidence: float = 0.0
    answerability: str = "unknown"

def _normalized_text(text: str) -> str:
    return " ".join(text.split())


def _is_model_timeout(error: Exception) -> bool:
    name = type(error).__name__.lower()
    message = str(error).lower()
    return isinstance(error, TimeoutError) or "timeout" in name or "timeout" in message


def _evidence_confidence(chunks: list[EvidenceChunk]) -> float:
    if not chunks:
        return 0.0
    if len(chunks) < 2:
        return 0.35
    return 0.85 if any(chunk.page_number > 0 for chunk in chunks) else 0.65


def _escape_boundary_markers(text: str) -> str:
    replacements = {
        "BEGIN UNTRUSTED EVIDENCE DATA": "[ESCAPED BEGIN EVIDENCE MARKER]",
        "END UNTRUSTED EVIDENCE DATA": "[ESCAPED END EVIDENCE MARKER]",
        "ALLOWED CHUNK IDS:": "[ESCAPED ALLOWED IDS MARKER]",
    }
    for marker, replacement in replacements.items():
        text = text.replace(marker, replacement)
    return text


def _validate_claims(
    claims: list[AnswerClaim], chunks: list[EvidenceChunk]
) -> list[AnswerClaim]:
    """Keep only claims whose citations can be checked against local chunks."""
    canonical = {chunk.chunk_id: chunk for chunk in chunks}
    validated: list[AnswerClaim] = []
    for claim in claims:
        if not claim.text.strip():
            continue
        citations: list[AnswerCitation] = []
        seen_chunks: set[str] = set()
        for citation in claim.citations:
            chunk = canonical.get(citation.chunk_id)
            quote = _normalized_text(citation.quote)
            if (
                chunk is None
                or citation.chunk_id in seen_chunks
                or citation.paper_id != chunk.paper_id
                or not quote
                or quote not in _normalized_text(chunk.text)
            ):
                continue
            seen_chunks.add(citation.chunk_id)
            citations.append(
                AnswerCitation(
                    chunk_id=chunk.chunk_id,
                    paper_id=chunk.paper_id,
                    title=chunk.title,
                    page_number=chunk.page_number,
                    quote=citation.quote,
                )
            )
        papers = {citation.paper_id for citation in citations}
        if not citations or (claim.kind == "synthesis" and len(papers) < 2):
            continue
        validated.append(AnswerClaim(text=claim.text, kind=claim.kind, citations=citations))
    return validated


class CitedQaService:
    """Generate an answer and deterministically validate every citation."""

    def __init__(
        self,
        retriever: Retriever,
        model,
        *,
        chunk_store: ChunkStore,
        answer_k: int = 8,
    ) -> None:
        self.retriever = retriever
        self.model = model
        self.chunk_store = chunk_store
        self.answer_k = answer_k
        self.last_trace = QATrace(status="not_started")
        self.last_search_audit = SearchAudit()
        self.last_review = EvidenceAssessment()
        self.last_context_audit = ContextAudit()

    def answer(self, question: str, *, paper_ids: list[str] | None = None,
               supplemental_queries: list[str] | None = None) -> CitedAnswer:
        started = perf_counter()
        self.last_search_audit = SearchAudit()
        self.last_review = EvidenceAssessment()
        self.last_context_audit = ContextAudit()
        retrieval_started = started
        try:
            # Keep English/general questions on the existing single-query path.
            # Domain questions written in CJK use the bounded deterministic plan
            # so aliases such as 叶绿素 -> chlorophyll-a can reach the index.
            plan = plan_query(question)
            result = retrieve_evidence(
                self.retriever, question, k=self.answer_k, paper_ids=paper_ids,
                supplemental_queries=supplemental_queries,
                max_queries=4,
            )
            self.last_search_audit = result.audit
            retrieved = result.chunks
            chunk_ids = list(dict.fromkeys(chunk.chunk_id for chunk in retrieved))
            chunks = self.chunk_store.get_chunks_by_ids(chunk_ids)
            if paper_ids is not None:
                chunks = [chunk for chunk in chunks if chunk.paper_id in paper_ids]
            # Complex comparisons get one specialist review and at most two new searches.
            if needs_evidence_review(question):
                self.last_review = review_evidence(self.model, question, chunks)
                new_queries = list(dict.fromkeys(q.strip()[:4000] for q in self.last_review.supplemental_queries if q.strip()))[:2]
                if new_queries:
                    try:
                        extra = retrieve_evidence(
                            self.retriever, new_queries[0], k=self.answer_k, paper_ids=paper_ids,
                            supplemental_queries=new_queries[1:], max_queries=len(new_queries),
                        )
                        # Interleave rounds so missing-aspect evidence survives the context budget.
                        combined = []
                        for i in range(max(len(retrieved), len(extra.chunks))):
                            combined.extend(items[i] for items in (retrieved, extra.chunks) if i < len(items))
                        retrieved = list({c.chunk_id: c for c in combined}.values())[:self.answer_k]
                        chunk_ids = [c.chunk_id for c in retrieved]
                        self.last_search_audit = merge_audits(result.audit, extra.audit, chunk_ids)
                        chunks = self.chunk_store.get_chunks_by_ids(chunk_ids)
                    except Exception:
                        pass  # A failed supplemental round preserves the initial evidence.
            chunks, self.last_context_audit = assemble_context(
                self.chunk_store, chunk_ids, paper_ids=paper_ids,
                max_chunks=self.answer_k * 3, canonical_chunks=chunks,
            )
            retrieval_confidence = float(
                getattr(getattr(self.retriever, "last_trace", None), "retrieval_confidence", 0.0)
            ) or (0.8 if retrieved else 0.0)
        except Exception:
            self.last_trace = QATrace(
                status="retrieval_failed",
                retrieval_ms=round((perf_counter() - retrieval_started) * 1000, 3),
                original_query=question,
                query_variants=list(plan.queries) if "plan" in locals() else [question],
                intent_confidence=plan.confidence if "plan" in locals() else 0.0,
                answerability="retrieval_failed",
            )
            return self._insufficient(question, level="none", reason="all retrieval branches failed")
        retrieval_ms = (perf_counter() - retrieval_started) * 1000
        if not chunks:
            self.last_trace = QATrace(
                status="insufficient_chunks",
                retrieval_ms=round(retrieval_ms, 3),
                retrieved_chunks=len(retrieved),
                canonical_chunks=len(chunks),
                original_query=question,
                query_variants=list(plan.queries),
                intent_confidence=plan.confidence,
                retrieval_confidence=retrieval_confidence,
                evidence_confidence=_evidence_confidence(chunks),
                answerability="insufficient_evidence",
            )
            return self._insufficient(question, level="none", reason="no canonical evidence chunks are available")

        request = [
            SystemMessage(content=load_cited_qa_prompt()),
            HumanMessage(content=self._render_request(question, chunks)),
        ]
        model_started = perf_counter()
        try:
            consume("model_calls")
            proposed_raw = self.model.with_structured_output(CitedAnswer).invoke(
                request
            )
            proposed = CitedAnswer.model_validate(proposed_raw)
        except Exception as exc:
            if _is_model_timeout(exc):
                answer = self._evidence_fallback(chunks)
                self.last_trace = QATrace(
                    status="model_timeout_fallback" if answer.evidence_sufficient else "model_timeout_insufficient",
                    retrieval_ms=round(retrieval_ms, 3),
                    model_ms=round((perf_counter() - model_started) * 1000, 3),
                    retrieved_chunks=len(retrieved),
                    canonical_chunks=len(chunks),
                    citation_count=len(answer.citations),
                    original_query=question,
                    query_variants=list(plan.queries),
                    intent_confidence=plan.confidence,
                    retrieval_confidence=retrieval_confidence,
                    evidence_confidence=_evidence_confidence(chunks),
                    answerability="answerable" if answer.evidence_sufficient else "insufficient_evidence",
                )
                return answer
            self.last_trace = QATrace(
                status="model_failed",
                retrieval_ms=round(retrieval_ms, 3),
                model_ms=round((perf_counter() - model_started) * 1000, 3),
                retrieved_chunks=len(retrieved),
                canonical_chunks=len(chunks),
                original_query=question,
                query_variants=list(plan.queries),
                intent_confidence=plan.confidence,
                retrieval_confidence=retrieval_confidence,
                evidence_confidence=_evidence_confidence(chunks),
                answerability="model_failed",
            )
            return self._insufficient(question, level="related", reason="answer generation failed")
        if not proposed.evidence_sufficient:
            self.last_trace = QATrace(
                status="model_marked_insufficient",
                retrieval_ms=round(retrieval_ms, 3),
                model_ms=round((perf_counter() - model_started) * 1000, 3),
                retrieved_chunks=len(retrieved),
                canonical_chunks=len(chunks),
                original_query=question,
                query_variants=list(plan.queries),
                intent_confidence=plan.confidence,
                retrieval_confidence=retrieval_confidence,
                evidence_confidence=_evidence_confidence(chunks),
                answerability="insufficient_evidence",
            )
            return self._insufficient(question, level="weak", reason="the model marked the retrieved evidence as insufficient")

        claims = proposed.claims
        if not claims and proposed.citations:
            claims = [
                AnswerClaim(
                    text=proposed.answer_markdown,
                    kind="direct",
                    citations=proposed.citations,
                )
            ]
        surviving_claims = _validate_claims(claims, chunks)
        if not surviving_claims:
            self.last_trace = QATrace(
                status="citation_validation_failed",
                retrieval_ms=round(retrieval_ms, 3),
                model_ms=round((perf_counter() - model_started) * 1000, 3),
                retrieved_chunks=len(retrieved),
                canonical_chunks=len(chunks),
                original_query=question,
                query_variants=list(plan.queries),
                intent_confidence=plan.confidence,
                retrieval_confidence=retrieval_confidence,
                evidence_confidence=_evidence_confidence(chunks),
                answerability="citation_validation_failed",
            )
            return self._insufficient(question, level="weak", reason="candidate citations did not match canonical chunks")
        surviving: list[AnswerCitation] = []
        seen: set[str] = set()
        for claim in surviving_claims:
            for citation in claim.citations:
                if citation.chunk_id not in seen:
                    seen.add(citation.chunk_id)
                    surviving.append(citation)
        answer = CitedAnswer(
            answer_markdown="\n\n".join(claim.text for claim in surviving_claims),
            claims=surviving_claims,
            citations=surviving,
            evidence_sufficient=True,
            evidence_level="direct",
            evidence_reason="citations matched canonical chunks; semantic support is not independently verified",
        )
        self.last_trace = QATrace(
            status="answered",
            retrieval_ms=round(retrieval_ms, 3),
            model_ms=round((perf_counter() - model_started) * 1000, 3),
            retrieved_chunks=len(retrieved),
            canonical_chunks=len(chunks),
            citation_count=len(answer.citations),
            original_query=question,
            query_variants=list(plan.queries),
            intent_confidence=plan.confidence,
            retrieval_confidence=retrieval_confidence,
            evidence_confidence=_evidence_confidence(chunks),
            answerability="answerable",
        )
        return answer

    @staticmethod
    def _evidence_fallback(chunks: list[EvidenceChunk]) -> CitedAnswer:
        """Return only canonical excerpts when the model cannot produce valid JSON citations."""
        claims: list[AnswerClaim] = []
        citations: list[AnswerCitation] = []
        for chunk in chunks[:4]:
            quote = " ".join(chunk.text.split())[:360]
            if not quote:
                continue
            citation = AnswerCitation(
                chunk_id=chunk.chunk_id,
                paper_id=chunk.paper_id,
                title=chunk.title,
                page_number=chunk.page_number,
                quote=quote,
            )
            citations.append(citation)
            claims.append(
                AnswerClaim(
                    text=f"《{chunk.title}》的相关证据：{quote}",
                    kind="direct",
                    citations=[citation],
                )
            )
        if len(claims) < 2:
            return CitedQaService._insufficient("", level="related", reason="model unavailable and too few usable excerpts for an extractive fallback")
        return CitedAnswer(
            answer_markdown="\n\n".join(claim.text for claim in claims),
            claims=claims,
            citations=citations,
            evidence_sufficient=True,
            evidence_level="direct",
            evidence_reason="canonical excerpts were returned after model fallback",
        )

    @staticmethod
    def _render_request(question: str, chunks: list[EvidenceChunk]) -> str:
        allowed = ", ".join(
            _escape_boundary_markers(chunk.chunk_id) for chunk in chunks
        )
        serialized = []
        for number, chunk in enumerate(chunks, start=1):
            serialized.append(
                json.dumps(
                    {
                        "number": number,
                        "chunk_id": _escape_boundary_markers(chunk.chunk_id),
                        "paper_id": _escape_boundary_markers(chunk.paper_id),
                        "title": _escape_boundary_markers(chunk.title),
                        "page_number": chunk.page_number,
                        "section": (
                            _escape_boundary_markers(chunk.section)
                            if chunk.section is not None
                            else None
                        ),
                        "text": _escape_boundary_markers(chunk.text),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        return (
            f"QUESTION: {_escape_boundary_markers(question)}\n"
            f"ALLOWED CHUNK IDS: {allowed}\n"
            "BEGIN UNTRUSTED EVIDENCE DATA\n"
            + "\n".join(serialized)
            + "\nEND UNTRUSTED EVIDENCE DATA"
        )

    @staticmethod
    def _insufficient(question: str, *, level: str = "related", reason: str | None = None) -> CitedAnswer:
        normalized = " ".join(question.split()) or "water color remote sensing prediction"
        return CitedAnswer(
            answer_markdown="Evidence is insufficient to support a reliable answer.",
            citations=[],
            evidence_sufficient=False,
            suggested_search_query=f"{normalized} supporting literature",
            evidence_level=level,
            evidence_reason=reason or "retrieval did not produce enough directly citable evidence",
        )
