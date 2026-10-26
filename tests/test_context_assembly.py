from domain.models import EvidenceChunk
from retrieval.context import assemble_context
from tests.test_qa import FailingModel, Retriever
from workflows.qa import CitedQaService


def chunk(key, paper="p1"):
    return EvidenceChunk(chunk_id=key, paper_id=paper, title=paper, page_number=1,
                         text=f"Canonical evidence about remote sensing {key}.")


def test_direct_hits_survive_neighbor_order_and_global_context_limit():
    anchors = [chunk("z-hit", "z-paper"), chunk("a-hit", "a-paper")]

    class Store:
        def get_chunks_by_ids(self, ids):
            return list(reversed(anchors))

        def get_chunks_with_context(self, ids, window):
            return [*(chunk(f"noise-{i}", "a-paper") for i in range(10)), *anchors]

    service = CitedQaService(Retriever(anchors), FailingModel(), chunk_store=Store(), answer_k=2)
    answer = service.answer("Which methods are present?")
    assert [c.chunk_id for c in answer.citations[:2]] == ["z-hit", "a-hit"]
    assert service.last_context_audit.canonical_anchors == 2
    assert service.last_context_audit.neighbor_chunks == 4


def test_optional_neighbor_failure_keeps_canonical_evidence():
    anchors = [chunk("c1"), chunk("c2")]

    class Store:
        def get_chunks_by_ids(self, ids):
            return anchors

        def get_chunks_with_context(self, ids, window):
            raise RuntimeError("private provider failure")

    service = CitedQaService(Retriever(anchors), FailingModel(), chunk_store=Store())
    answer = service.answer("Which methods are present?")
    assert answer.evidence_sufficient
    assert {c.chunk_id for c in answer.citations} == {"c1", "c2"}
    assert service.last_context_audit.neighbor_load_failed


def test_scope_unrequested_ids_and_stale_anchor_payload_cannot_enter_context():
    canonical = chunk("anchor")

    class Store:
        def get_chunks_by_ids(self, ids):
            return [canonical, chunk("foreign", "foreign-paper"), chunk("unexpected")]

        def get_chunks_with_context(self, ids, window):
            assert ids == ["anchor"]
            return [canonical.model_copy(update={"text": "stale text"}),
                    chunk("foreign", "foreign-paper"), chunk("neighbor")]

    result, audit = assemble_context(Store(), ["anchor", "missing"], paper_ids=["p1"], max_chunks=3)
    assert [c.chunk_id for c in result] == ["anchor", "neighbor"]
    assert result[0].text == canonical.text
    assert audit.missing_anchors == 1


def test_empty_scope_does_not_read_store():
    result, audit = assemble_context(object(), ["anchor"], paper_ids=[])
    assert result == []
    assert audit.canonical_anchors == 0
