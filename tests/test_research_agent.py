from __future__ import annotations

from domain.models import (
    AnswerCitation,
    CitedAnswer,
    EvidenceChunk,
    ResearchPlan,
    ResearchPlanFinding,
    ResearchConversationMessage,
    ResearchConversationState,
)
from agent.research_agent import ResearchAgentService
from retrieval.hybrid import RetrievalTrace
from workflows.qa import QATrace
from workflows.research_plan import ResearchPlanService


class StructuredModel:
    def __init__(self, payload):
        self.payload = payload

    def with_structured_output(self, schema):
        payload = self.payload

        class Bound:
            def invoke(self, messages):
                return payload

        return Bound()


class GeneralChatModel(StructuredModel):
    def __init__(self, payload, answer):
        super().__init__(payload)
        self.answer = answer

    def invoke(self, messages):
        return type("Response", (), {"content": self.answer})()


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, *, k, paper_ids=None):
        return self.chunks[:k]


class FakeStore:
    def __init__(self, chunks):
        self.by_id = {chunk.chunk_id: chunk for chunk in chunks}

    def get_chunks_by_ids(self, chunk_ids):
        return [self.by_id[item] for item in chunk_ids if item in self.by_id]


class FakeQa:
    def __init__(self):
        self.queries = []

    def answer(self, query):
        self.queries.append(query)
        return CitedAnswer(
            answer_markdown="Sentinel-2 is supported.",
            citations=[
                AnswerCitation(
                    chunk_id="c1",
                    paper_id="p1",
                    title="Paper One",
                    page_number=2,
                    quote="Sentinel-2 imagery",
                )
            ],
            evidence_sufficient=True,
        )


class FakePlan:
    def __init__(self):
        self.queries = []

    def plan(self, query):
        self.queries.append(query)
        citation = AnswerCitation(
            chunk_id="c1",
            paper_id="p1",
            title="Paper One",
            page_number=2,
            quote="Sentinel-2 imagery",
        )
        return ResearchPlan(
            topic=query,
            findings=[ResearchPlanFinding(text="A supported finding", citations=[citation])],
            suggested_steps=["define a reproducible baseline"],
            evidence_sufficient=True,
        )


def _chunks():
    return [
        EvidenceChunk(
            chunk_id="c1",
            paper_id="p1",
            title="Paper One",
            page_number=2,
            text="The study used Sentinel-2 imagery for chlorophyll-a prediction.",
        ),
        EvidenceChunk(
            chunk_id="c2",
            paper_id="p2",
            title="Paper Two",
            page_number=4,
            text="Random forest was evaluated against a linear baseline.",
        ),
    ]


def test_research_plan_keeps_only_canonical_chunk_quotes():
    chunks = _chunks()
    model = StructuredModel(
        {
            "findings": [
                {
                    "text": "supported",
                    "citations": [
                        {
                            "chunk_id": "c1",
                            "paper_id": "invented",
                            "title": "invented",
                            "page_number": 99,
                            "quote": "Sentinel-2 imagery",
                        }
                    ],
                },
                {
                    "text": "unsupported",
                    "citations": [
                        {
                            "chunk_id": "c2",
                            "paper_id": "p2",
                            "title": "Paper Two",
                            "page_number": 4,
                            "quote": "a quote that is not present",
                        }
                    ],
                },
            ],
            "suggested_steps": ["compare baselines"],
            "evidence_sufficient": True,
        }
    )
    plan = ResearchPlanService(
        FakeRetriever(chunks), model, chunk_store=FakeStore(chunks)
    ).plan("chlorophyll prediction")

    assert plan.evidence_sufficient
    assert [item.text for item in plan.findings] == ["supported"]
    citation = plan.findings[0].citations[0]
    assert (citation.paper_id, citation.title, citation.page_number) == (
        "p1",
        "Paper One",
        2,
    )


def test_agent_routes_one_tool_and_persists_explicit_research_scope():
    qa, plan = FakeQa(), FakePlan()
    model = StructuredModel(
        {
            "action": "research_plan",
            "rewritten_query": "Sentinel-2 lake chlorophyll-a research plan after 2022",
            "scope_updates": {
                "topic": "lake chlorophyll-a prediction",
                "prediction_target": "chlorophyll-a",
                "sensor": "Sentinel-2",
                "year_range": "after 2022",
                "method_constraints": ["machine learning"],
            },
        }
    )
    service = ResearchAgentService(model=model, qa_service=qa, plan_service=plan)
    turn = service.chat("我想用 Sentinel-2 研究湖泊叶绿素 a，应该从哪里开始？")

    assert qa.queries == []
    assert plan.queries == ["Sentinel-2 lake chlorophyll-a research plan after 2022"]
    assert turn.reply.tool_name == "research_plan"
    assert turn.state.scope.sensor == "Sentinel-2"
    assert turn.state.scope.method_constraints == ["machine learning"]


def test_agent_keeps_general_chat_outside_local_evidence_tools():
    model = GeneralChatModel(
        {
            "skill_id": "general_chat",
            "rewritten_query": "hello",
            "scope_updates": {},
        },
        "你好，我可以继续普通对话。",
    )
    qa, plan = FakeQa(), FakePlan()

    turn = ResearchAgentService(model=model, qa_service=qa, plan_service=plan).chat("你好")

    assert turn.reply.tool_name == "general_chat"
    assert turn.reply.content == "你好，我可以继续普通对话。"
    assert turn.reply.citations == []
    assert qa.queries == []
    assert plan.queries == []


def test_agent_falls_back_to_uncited_explanation_when_local_evidence_is_insufficient():
    class EmptyQa:
        def answer(self, query):
            return CitedAnswer(
                answer_markdown="",
                evidence_sufficient=False,
                suggested_search_query="叶绿素是什么 supporting literature",
            )

    model = GeneralChatModel(
        {
            "skill_id": "evidence_qa",
            "rewritten_query": "叶绿素是什么",
            "scope_updates": {},
        },
        "叶绿素是植物和藻类进行光合作用的重要色素。",
    )
    turn = ResearchAgentService(model=model, qa_service=EmptyQa(), plan_service=FakePlan()).chat(
        "叶绿素是什么"
    )

    assert turn.reply.tool_name == "general_chat"
    assert "概念解释" in turn.reply.content or "本地知识库未找到足够" in turn.reply.content
    assert "叶绿素是植物" in turn.reply.content
    assert turn.reply.citations == []


def test_agent_falls_back_to_uncited_plan_when_local_plan_evidence_is_insufficient():
    class EmptyPlan:
        def plan(self, query):
            return ResearchPlan(
                topic=query,
                findings=[],
                suggested_steps=[],
                evidence_sufficient=False,
                suggested_search_query="Sentinel-2 chlorophyll baseline",
            )

    model = GeneralChatModel(
        {
            "skill_id": "research_plan",
            "rewritten_query": "Sentinel-2 chlorophyll baseline",
            "scope_updates": {},
        },
        "建议先定义预测目标、数据切分、基线模型和评价指标。",
    )
    turn = ResearchAgentService(model=model, qa_service=FakeQa(), plan_service=EmptyPlan()).chat(
        "如何设计一个叶绿素预测基线？"
    )

    assert turn.reply.tool_name == "general_chat"
    assert "通用研究建议" in turn.reply.content
    assert "定义预测目标" in turn.reply.content


def test_agent_planner_overrides_model_route_for_concept_question():
    model = GeneralChatModel(
        {
            "skill_id": "evidence_qa",
            "rewritten_query": "叶绿素是什么",
            "scope_updates": {},
        },
        "叶绿素是光合作用色素。",
    )

    turn = ResearchAgentService(model=model, qa_service=FakeQa(), plan_service=FakePlan()).chat(
        "叶绿素是什么"
    )

    assert turn.reply.tool_name == "general_chat"
    assert turn.reply.citations == []


def test_agent_reply_contains_safe_runtime_diagnostics():
    qa, plan = FakeQa(), FakePlan()
    qa.last_trace = QATrace(
        status="answered",
        retrieval_ms=3.5,
        model_ms=8.25,
        retrieved_chunks=6,
        canonical_chunks=5,
        citation_count=2,
    )
    qa.retriever = type(
        "Retriever",
        (),
        {
            "last_trace": RetrievalTrace(
                query="secret query must not be returned",
                keyword_candidates=10,
                vector_candidates=10,
                fused_candidates=14,
                selected_count=5,
                selected_chunk_ids=["c1"],
                selected_paper_ids=["p1"],
                latency_ms=3.5,
            )
        },
    )()
    model = StructuredModel(
        {
            "action": "evidence_qa",
            "rewritten_query": "safe rewritten query",
            "scope_updates": {},
        }
    )

    turn = ResearchAgentService(model=model, qa_service=qa, plan_service=plan).chat("question")

    diagnostics = turn.reply.diagnostics
    assert diagnostics is not None
    assert diagnostics.skill_id == "evidence_qa"
    assert diagnostics.route_mode == "model"
    assert diagnostics.route_reason == "model_structured_route"
    assert diagnostics.skill_version == "1.0"
    assert diagnostics.evidence_sufficient is True
    assert diagnostics.retrieval_candidates == 20
    assert diagnostics.evidence_chunks == 5
    assert diagnostics.citation_count == 2
    assert diagnostics.retrieval_ms == 3.5
    assert diagnostics.model_ms == 8.25
    assert diagnostics.total_ms >= diagnostics.model_ms
    assert "secret" not in diagnostics.model_dump_json()
    assert turn.state.messages[-1].diagnostics == diagnostics


def test_agent_diagnostics_mark_router_fallback_and_tolerate_missing_traces():
    class InvalidModel:
        def with_structured_output(self, schema):
            class Bound:
                def invoke(self, messages):
                    raise RuntimeError("model output contains a secret URL")

            return Bound()

    turn = ResearchAgentService(
        model=InvalidModel(), qa_service=FakeQa(), plan_service=FakePlan()
    ).chat("下一步怎么做")

    diagnostics = turn.reply.diagnostics
    assert diagnostics is not None
    assert diagnostics.route_mode == "fallback"
    assert diagnostics.skill_id == "research_plan"
    assert diagnostics.route_reason == "fallback_rule_research_plan"
    assert diagnostics.skill_version == "1.0"
    assert diagnostics.evidence_sufficient is True
    assert diagnostics.retrieval_candidates == 0
    assert diagnostics.evidence_chunks == 0
    assert diagnostics.citation_count == 1
    assert "secret URL" not in diagnostics.model_dump_json()


def test_agent_compacts_old_dialogue_and_keeps_recent_eight_messages():
    qa, plan = FakeQa(), FakePlan()
    model = StructuredModel(
        {
            "action": "evidence_qa",
            "rewritten_query": "chlorophyll-a",
            "scope_updates": {},
        }
    )
    service = ResearchAgentService(model=model, qa_service=qa, plan_service=plan)
    conversation = None
    for index in range(6):
        conversation = service.chat(f"question {index}", conversation).state

    assert len(conversation.messages) == 8
    assert "question 0" in conversation.summary
    assert conversation.messages[-1].tool_name == "evidence_qa"


def test_agent_context_preserves_recent_citation_ids_and_policy():
    captured = []

    class Model:
        def with_structured_output(self, schema):
            class Bound:
                def invoke(self, messages):
                    captured.append(messages[1].content)
                    return {"skill_id": "evidence_qa", "rewritten_query": "follow up"}

            return Bound()

    class Qa:
        def answer(self, query):
            return CitedAnswer(answer_markdown="supported", evidence_sufficient=True)

    citation = AnswerCitation(
        chunk_id="paper-1:p1:c0",
        paper_id="paper-1",
        title="Paper",
        page_number=1,
        quote="evidence",
    )
    conversation = ResearchConversationState(
        messages=[
            ResearchConversationMessage(
                role="assistant", content="earlier", citations=[citation]
            )
        ]
    )
    ResearchAgentService(model=Model(), qa_service=Qa(), plan_service=Qa()).chat(
        "next", conversation
    )

    assert '"chunk_id": "paper-1:p1:c0"' in captured[0]
    assert '"evidence_required": true' in captured[0]
