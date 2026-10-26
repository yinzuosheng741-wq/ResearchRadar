import pytest

from domain.models import EvidenceChunk
from retrieval.evidence_requirements import evidence_requirement_gap
from workflows.qa import CitedQaService


def chunk(page=0):
    return EvidenceChunk(chunk_id="abstract", paper_id="paper", title="Water study", page_number=page,
                         text="The abstract reports a remote sensing model and an overall evaluation result.")


@pytest.mark.parametrize("question", ["学习率和超参数是多少？", "解释表 2 的每一列", "What is the learning rate?",
                                      "Show equation 3", "请核对训练与测试的划分比例"])
def test_abstract_cannot_support_explicit_body_detail_requests(question):
    assert evidence_requirement_gap(question, [chunk()]) == "abstract_only_for_fulltext_detail"


@pytest.mark.parametrize("question", ["摘要说明使用了什么方法？", "Summarize the main conclusion."])
def test_abstract_can_still_be_used_for_high_level_questions(question):
    assert evidence_requirement_gap(question, [chunk()]) is None


def test_body_evidence_is_not_automatically_declared_insufficient_or_sufficient():
    assert evidence_requirement_gap("表 2 如何解释？", [chunk(2)]) is None


def test_qa_stops_before_answer_generation_when_only_abstract_supports_table_request():
    class Store:
        def get_chunks_by_ids(self, _ids):
            return [chunk()]

    class Retriever:
        def search(self, *_args, **_kwargs):
            return [chunk()]

    class Model:
        def with_structured_output(self, _schema):
            raise AssertionError("answer model must not be called for missing fulltext")

    service = CitedQaService(Retriever(), Model(), chunk_store=Store())
    answer = service.answer("解释表 2 的各列")
    assert not answer.evidence_sufficient
    assert answer.citations == []
    assert "摘要" in answer.answer_markdown
    assert service.last_trace.status == "abstract_only_for_fulltext_detail"
