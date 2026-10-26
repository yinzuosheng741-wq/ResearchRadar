from __future__ import annotations

from domain.models import EvidenceChunk
from workflows.qa import CitedQaService, QATrace


class FailingModel:
    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        raise TimeoutError("structured_output_timeout")


class Retriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, *, k, paper_ids=None):
        return self.chunks[:k]


class RecordingRetriever(Retriever):
    def __init__(self, chunks):
        super().__init__(chunks)
        self.queries = []

    def search(self, query, *, k, paper_ids=None):
        self.queries.append(query)
        return super().search(query, k=k, paper_ids=paper_ids)


class Store:
    def __init__(self, chunks):
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}

    def get_chunks_by_ids(self, chunk_ids):
        return [self.chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in self.chunks]


def _chunk(number: int) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=f"c{number}",
        paper_id=f"p{number}",
        title=f"Paper {number}",
        page_number=number,
        text=f"Canonical evidence passage {number} about remote sensing methods.",
    )


def test_qa_falls_back_to_canonical_local_evidence_when_structured_model_fails():
    chunks = [_chunk(1), _chunk(2)]
    service = CitedQaService(Retriever(chunks), FailingModel(), chunk_store=Store(chunks))
    answer = service.answer("Which methods are present?")

    assert answer.evidence_sufficient is True
    assert [citation.chunk_id for citation in answer.citations] == ["c1", "c2"]
    assert "Canonical evidence passage 1" in answer.answer_markdown
    assert all(claim.kind == "direct" for claim in answer.claims)
    assert isinstance(service.last_trace, QATrace)
    assert service.last_trace.status == "model_timeout_fallback"
    assert service.last_trace.retrieved_chunks == 2
    assert service.last_trace.intent_confidence > 0
    assert service.last_trace.evidence_confidence > 0
    assert service.last_trace.answerability == "answerable"
    assert service.last_trace.query_variants
    assert answer.evidence_level == "direct"


def test_qa_timeout_with_one_chunk_stays_insufficient_without_false_answerable_trace():
    chunk = _chunk(1)
    service = CitedQaService(Retriever([chunk]), FailingModel(), chunk_store=Store([chunk]))
    answer = service.answer("Which methods are present?")

    assert answer.evidence_sufficient is False
    assert answer.citations == []
    assert service.last_trace.status == "model_timeout_insufficient"
    assert service.last_trace.answerability == "insufficient_evidence"
    assert answer.evidence_level == "related"


def test_one_canonical_chunk_can_support_a_valid_direct_answer():
    chunk = _chunk(1)

    class Model:
        def with_structured_output(self, schema):
            return self

        def invoke(self, messages):
            return {
                "answer_markdown": "The passage describes remote sensing methods.",
                "evidence_sufficient": True,
                "claims": [{
                    "text": "The passage describes remote sensing methods.", "kind": "direct",
                    "citations": [{"chunk_id": chunk.chunk_id, "paper_id": chunk.paper_id,
                                   "title": chunk.title, "page_number": chunk.page_number,
                                   "quote": chunk.text}],
                }],
            }

    service = CitedQaService(Retriever([chunk]), Model(), chunk_store=Store([chunk]))
    answer = service.answer("Which methods are present?")
    assert answer.evidence_sufficient
    assert [c.chunk_id for c in answer.citations] == [chunk.chunk_id]
    assert service.last_trace.status == "answered"


def test_qa_uses_bounded_mult_query_plan_for_chinese_domain_question():
    chunks = [_chunk(1), _chunk(2)]
    retriever = RecordingRetriever(chunks)
    service = CitedQaService(retriever, FailingModel(), chunk_store=Store(chunks))

    answer = service.answer("叶绿素如何用遥感预测？")

    assert answer.evidence_sufficient is True
    assert len(retriever.queries) >= 2
    assert any("chlorophyll-a" in query for query in retriever.queries)
    assert any(
        "chlorophyll-a" in query
        and "remote sensing" in query
        and "prediction" in query
        for query in retriever.queries[:4]
    )
