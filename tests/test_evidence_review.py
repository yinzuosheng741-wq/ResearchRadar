from agent.execution import ExecutionContext, execution_scope
from domain.models import CitedAnswer, EvidenceChunk
from workflows.qa import CitedQaService


def test_comparison_reviewer_can_recover_missing_side_within_request_budget():
    chunks = {key: EvidenceChunk(chunk_id=key, paper_id=key, title=key, page_number=1,
                                text=f"Evidence about {key}") for key in ("sentinel", "landsat")}

    class Retriever:
        def search(self, query, *, k, paper_ids=None):
            return [chunks["landsat"]] if query == "landsat missing validation" else [chunks["sentinel"]]

    class Store:
        def get_chunks_by_ids(self, ids):
            return [chunks[key] for key in ids]

    class Model:
        def with_structured_output(self, schema):
            self.schema = schema
            return self

        def invoke(self, messages):
            if self.schema.__name__ == "EvidenceAssessment":
                return {"missing_aspects": ["Landsat validation"], "supplemental_queries": ["landsat missing validation"]}
            raise TimeoutError()

    service = CitedQaService(Retriever(), Model(), chunk_store=Store(), answer_k=2)
    context = ExecutionContext()
    with execution_scope(context):
        answer = service.answer("compare Sentinel and Landsat")
    assert {c.chunk_id for c in answer.citations} == {"sentinel", "landsat"}
    assert context.search_calls == 2
    assert service.last_search_audit.attempted == context.search_calls
    assert service.last_search_audit.unique_candidates == 2
    assert context.model_calls == 2
    assert service.last_review.missing_aspects == ["Landsat validation"]


def test_zero_model_budget_does_not_call_provider():
    from agent.research_agent import ResearchAgentService

    class Model:
        def with_structured_output(self, schema):
            raise AssertionError("budget must stop provider call")

        def invoke(self, messages):
            raise AssertionError("budget must stop provider call")

    result = ResearchAgentService(model=Model(), qa_service=None, plan_service=None).chat(
        "hello", execution_context=ExecutionContext(max_model_calls=0)
    )
    assert not result.reply.evidence_sufficient
    assert "预算" in result.reply.content
