"""Evidence-validated research starting plans."""

from __future__ import annotations

import json
from typing import Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from domain.models import (
    AnswerCitation,
    EvidenceChunk,
    ResearchPlan,
    ResearchPlanFinding,
)
from utils.prompt_loader import load_research_plan_prompt
from agent.execution import consume
from retrieval.coordinator import retrieve_evidence, SearchAudit


class Retriever(Protocol):
    def search(
        self, query: str, *, k: int, paper_ids: list[str] | None = None
    ) -> list[EvidenceChunk]: ...


class ChunkStore(Protocol):
    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[EvidenceChunk]: ...


class _DraftFinding(BaseModel):
    text: str
    citations: list[AnswerCitation] = Field(default_factory=list)


class _PlanDraft(BaseModel):
    findings: list[_DraftFinding] = Field(default_factory=list)
    suggested_steps: list[str] = Field(default_factory=list)
    evidence_sufficient: bool = False


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _escape_markers(text: str) -> str:
    for marker in (
        "BEGIN UNTRUSTED EVIDENCE DATA",
        "END UNTRUSTED EVIDENCE DATA",
        "ALLOWED CHUNK IDS:",
    ):
        text = text.replace(marker, "[ESCAPED EVIDENCE MARKER]")
    return text


class ResearchPlanService:
    def __init__(
        self, retriever: Retriever, model, *, chunk_store: ChunkStore, plan_k: int = 12
    ) -> None:
        self.retriever = retriever
        self.model = model
        self.chunk_store = chunk_store
        self.plan_k = plan_k
        self.last_search_audit = SearchAudit()

    def plan(self, topic: str, *, paper_ids: list[str] | None = None,
             supplemental_queries: list[str] | None = None) -> ResearchPlan:
        normalized_topic = _normalized(topic)
        if not normalized_topic:
            raise ValueError("research_plan_topic_required")
        try:
            result = retrieve_evidence(self.retriever, normalized_topic, k=self.plan_k,
                                       paper_ids=paper_ids, supplemental_queries=supplemental_queries)
            self.last_search_audit = result.audit
            retrieved = result.chunks
            ids = list(dict.fromkeys(item.chunk_id for item in retrieved))
            chunks = self.chunk_store.get_chunks_by_ids(ids)
            if paper_ids is not None:
                chunks = [chunk for chunk in chunks if chunk.paper_id in paper_ids]
        except Exception:
            return self._insufficient(normalized_topic)
        if len(chunks) < 2:
            return self._insufficient(normalized_topic)

        request = [
            SystemMessage(content=load_research_plan_prompt()),
            HumanMessage(content=self._request(normalized_topic, chunks)),
        ]
        try:
            consume("model_calls")
            raw = self.model.with_structured_output(_PlanDraft).invoke(request)
            draft = _PlanDraft.model_validate(raw)
        except Exception:
            return self._insufficient(normalized_topic)
        if not draft.evidence_sufficient:
            return self._insufficient(normalized_topic)

        canonical = {chunk.chunk_id: chunk for chunk in chunks}
        findings: list[ResearchPlanFinding] = []
        for proposed in draft.findings[:8]:
            citations: list[AnswerCitation] = []
            seen: set[str] = set()
            for citation in proposed.citations:
                chunk = canonical.get(citation.chunk_id)
                quote = _normalized(citation.quote)
                if (
                    chunk is None
                    or chunk.chunk_id in seen
                    or not quote
                    or quote not in _normalized(chunk.text)
                ):
                    continue
                seen.add(chunk.chunk_id)
                citations.append(
                    AnswerCitation(
                        chunk_id=chunk.chunk_id,
                        paper_id=chunk.paper_id,
                        title=chunk.title,
                        page_number=chunk.page_number,
                        quote=citation.quote,
                    )
                )
            if citations and proposed.text.strip():
                findings.append(
                    ResearchPlanFinding(text=proposed.text.strip(), citations=citations)
                )
        if not findings:
            return self._insufficient(normalized_topic)
        steps = [step.strip() for step in draft.suggested_steps[:8] if step.strip()]
        return ResearchPlan(
            topic=normalized_topic,
            findings=findings,
            suggested_steps=steps,
            evidence_sufficient=True,
        )

    @staticmethod
    def _request(topic: str, chunks: list[EvidenceChunk]) -> str:
        allowed = ", ".join(_escape_markers(chunk.chunk_id) for chunk in chunks)
        evidence = [
            json.dumps(
                {
                    "chunk_id": _escape_markers(chunk.chunk_id),
                    "paper_id": _escape_markers(chunk.paper_id),
                    "title": _escape_markers(chunk.title),
                    "page_number": chunk.page_number,
                    "text": _escape_markers(chunk.text),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            for chunk in chunks
        ]
        return (
            f"RESEARCH TOPIC: {_escape_markers(topic)}\n"
            f"ALLOWED CHUNK IDS: {allowed}\n"
            "BEGIN UNTRUSTED EVIDENCE DATA\n"
            + "\n".join(evidence)
            + "\nEND UNTRUSTED EVIDENCE DATA"
        )

    @staticmethod
    def _insufficient(topic: str) -> ResearchPlan:
        return ResearchPlan(
            topic=topic,
            evidence_sufficient=False,
            suggested_search_query=f"{topic} water quality remote sensing methods",
        )
