import json
import io
import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from evaluation.dataset import DatasetError, load_answer_rows, load_questions
from evaluation.metrics import (
    citation_precision,
    deterministic_mean,
    evidence_group_recall_at_k,
    evidence_coverage,
    recall_at_k,
    reciprocal_rank,
    unsupported_claim_rate,
)
from evaluation.metrics import answer_level_metrics
from evaluation.run import EvaluationError, _two_stage_acceptance, run_evaluation
from evaluation.health import run_provider_health
from domain.models import AnswerCitation, CitedAnswer, EvidenceChunk
from retrieval.hybrid import RetrievalTrace
from workflows.qa import QATrace
import app


def test_required_metric_examples():
    assert recall_at_k(["a", "b", "c"], {"b", "d"}, k=3) == 0.5
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert citation_precision(valid=3, total=4) == 0.75


def test_metrics_deduplicate_rankings_and_use_safe_zero_denominators():
    assert recall_at_k(["a", "a", "b"], ["a", "b"], k=2) == 0.5
    assert reciprocal_rank(["x", "x", "b"], ["b"]) == pytest.approx(1 / 3)
    assert recall_at_k(["a"], [], k=5) == 0.0


def test_evidence_group_recall_treats_chunks_within_a_group_as_alternatives():
    groups = [("p1:c1", "p1:c2"), ("p2:c1",)]

    assert evidence_group_recall_at_k(["p1:c2", "p2:c1"], groups, k=5) == 1.0
    assert evidence_group_recall_at_k(["p1:c1"], groups, k=5) == 0.5


def test_evidence_group_recall_deduplicates_ranked_ids_and_rejects_empty_groups():
    assert evidence_group_recall_at_k(["p1:c1", "p1:c1"], [("p1:c1",)], k=5) == 1.0

    with pytest.raises(ValueError, match="^evaluation_metric_invalid$"):
        evidence_group_recall_at_k(["p1:c1"], [()], k=5)


def test_dataset_accepts_optional_category_and_defaults_to_uncategorized(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "question": "q",
                "evidence_groups": [
                    {
                        "paper_id": "p1",
                        "chunk_ids": ["p1:c1"],
                        "rationale": "The passage directly supports the question.",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_questions(path)[0].category == "uncategorized"


def test_dataset_rejects_invalid_category_without_echoing_value(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "question": "q",
                "category": "自然语言",
                "evidence_groups": [
                    {
                        "paper_id": "p1",
                        "chunk_ids": ["p1:c1"],
                        "rationale": "The passage directly supports the question.",
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match="^evaluation_dataset_invalid_category$"):
        load_questions(path)
    assert citation_precision(0, 0) == 0.0
    assert evidence_coverage(0, 0) == 0.0
    assert unsupported_claim_rate(0, 0) == 0.0
    assert deterministic_mean([]) == 0.0
    assert deterministic_mean([0.25, 0.75]) == 0.5


@pytest.mark.parametrize(
    "operation",
    [
        lambda: recall_at_k([], [], k=0),
        lambda: citation_precision(-1, 1),
        lambda: citation_precision(2, 1),
        lambda: evidence_coverage(2, 1),
        lambda: unsupported_claim_rate(1, 0),
    ],
)
def test_metrics_reject_invalid_counts_and_k(operation):
    with pytest.raises(ValueError, match="^evaluation_metric_invalid$"):
        operation()


@pytest.mark.parametrize(
    "operation",
    [
        lambda: recall_at_k([1], ["a"], k=5),
        lambda: recall_at_k(["a"], [None], k=5),
        lambda: reciprocal_rank(["a", ""], ["a"]),
        lambda: reciprocal_rank(["a"], 42),
    ],
)
def test_retrieval_metrics_reject_non_string_or_empty_ids(operation):
    with pytest.raises(ValueError, match="^evaluation_metric_invalid$"):
        operation()


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _annotated(question_id="q01", question="Which sensor?"):
    return {
        "question_id": question_id,
        "question": question,
        "category": "exact_term",
        "evidence_groups": [
            {
                "paper_id": "paper-1",
                "chunk_ids": ["paper-1:p3:c1", "paper-1:p3:c2"],
                "rationale": "Either passage directly identifies the sensor.",
            }
        ],
    }


def test_dataset_parses_evidence_groups_and_derives_ordered_ids(tmp_path):
    path = tmp_path / "questions.jsonl"
    _write_jsonl(path, [_annotated()])

    question = load_questions(path)[0]

    assert question.relevant_paper_ids == ("paper-1",)
    assert question.relevant_chunk_ids == ("paper-1:p3:c1", "paper-1:p3:c2")
    assert question.evidence_groups[0].rationale.startswith("Either passage")


def test_dataset_derives_stable_id_for_missing_question_id(tmp_path):
    row = _annotated()
    del row["question_id"]
    path = tmp_path / "questions.jsonl"
    _write_jsonl(path, [row])

    first = load_questions(path)
    second = load_questions(path)

    assert first[0].question_id == second[0].question_id
    assert first[0].question_id.startswith("derived-")


@pytest.mark.parametrize(
    "contents,code",
    [
        ("{not json}\n", "evaluation_dataset_malformed_json"),
        (json.dumps({**_annotated(), "question": " "}) + "\n", "evaluation_dataset_invalid"),
        (
            json.dumps({**_annotated(), "evidence_groups": []}) + "\n",
            "evaluation_dataset_invalid_evidence_group",
        ),
        (
            json.dumps({**_annotated(), "question": "C:\\\\Users\\\\name\\\\secret"}) + "\n",
            "evaluation_dataset_sensitive_content",
        ),
        (
            json.dumps({**_annotated(), "question": "/home/name/private/file"}) + "\n",
            "evaluation_dataset_sensitive_content",
        ),
        (
            json.dumps({**_annotated(), "question": "contact private@example.test"}) + "\n",
            "evaluation_dataset_sensitive_content",
        ),
        (
            json.dumps({**_annotated(), "unknown": "value"}) + "\n",
            "evaluation_dataset_invalid",
        ),
    ],
)
def test_dataset_rejects_invalid_rows_without_echoing_content(tmp_path, contents, code):
    path = tmp_path / "questions.jsonl"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(DatasetError) as captured:
        load_questions(path)

    assert str(captured.value) == code
    assert "secret" not in str(captured.value)


@pytest.mark.parametrize(
    "groups,extra,code",
    [
        ([], {}, "evaluation_dataset_invalid_evidence_group"),
        (
            [{"paper_id": "paper-1", "chunk_ids": ["paper-1:p3:c1"], "rationale": " "}],
            {},
            "evaluation_dataset_invalid_evidence_group",
        ),
        (
            [{"paper_id": "paper-1", "chunk_ids": ["paper-1:p3:c1", "paper-1:p3:c1"], "rationale": "Direct support."}],
            {},
            "evaluation_dataset_invalid_evidence_group",
        ),
        (
            [{"paper_id": "paper-1", "chunk_ids": ["paper-2:p3:c1"], "rationale": "Direct support."}],
            {},
            "evaluation_dataset_invalid_evidence_group",
        ),
        (
            [
                {"paper_id": "paper-1", "chunk_ids": ["paper-1:p3:c1"], "rationale": "Direct support."},
                {"paper_id": "paper-1", "chunk_ids": ["paper-1:p3:c1"], "rationale": "Direct support."},
            ],
            {},
            "evaluation_dataset_invalid_evidence_group",
        ),
        (None, {"unexpected": "field"}, "evaluation_dataset_invalid"),
        (
            None,
            {"relevant_paper_ids": ["paper-1"], "relevant_chunk_ids": ["paper-1:p3:c1"]},
            "evaluation_dataset_invalid",
        ),
    ],
)
def test_dataset_rejects_invalid_evidence_group_contract_without_echoing_content(
    tmp_path, groups, extra, code
):
    row = _annotated()
    if groups is not None:
        row["evidence_groups"] = groups
    row.update(extra)
    path = tmp_path / "questions.jsonl"
    _write_jsonl(path, [row])

    with pytest.raises(DatasetError) as captured:
        load_questions(path)

    assert str(captured.value) == code
    assert "paper-2" not in str(captured.value)


def test_dataset_rejects_duplicate_question_ids_and_duplicate_rows(tmp_path):
    duplicate_id = tmp_path / "duplicate-id.jsonl"
    _write_jsonl(duplicate_id, [_annotated(), _annotated(question="Different?")])
    with pytest.raises(DatasetError, match="^evaluation_dataset_duplicate_question_id$"):
        load_questions(duplicate_id)

    duplicate_row = tmp_path / "duplicate-row.jsonl"
    row = _annotated()
    del row["question_id"]
    _write_jsonl(duplicate_row, [row, row])
    with pytest.raises(DatasetError, match="^evaluation_dataset_duplicate_row$"):
        load_questions(duplicate_row)


def test_normal_loader_rejects_placeholders_but_template_mode_loads_20_topics():
    path = Path("data/evaluation/questions.jsonl")
    with pytest.raises(DatasetError, match="^evaluation_dataset_unannotated$"):
        load_questions(path)

    questions = load_questions(path, template_mode=True)
    text = " ".join(item.question.casefold() for item in questions)
    assert len(questions) == 20
    assert len({item.question_id for item in questions}) == 20
    for topic in ("target", "sensor", "dataset", "model", "metric", "limitation", "temporal"):
        assert topic in text


def test_annotated_dataset_contains_concrete_local_evidence_ids():
    questions = load_questions(Path("data/evaluation/questions-annotated.jsonl"))
    assert len(questions) == 48
    assert {item.category for item in questions} == {
        "exact_term",
        "natural_language",
        "cross_paper",
    }
    assert {category: sum(q.category == category for q in questions) for category in {"exact_term", "natural_language", "cross_paper"}} == {
        "exact_term": 16,
        "natural_language": 16,
        "cross_paper": 16,
    }
    assert all(
        group.rationale.strip()
        for question in questions
        for group in question.evidence_groups
    )
    assert all(
        ":p" in chunk_id or ":abstract:" in chunk_id
        for question in questions
        for chunk_id in question.relevant_chunk_ids
    )
@pytest.mark.parametrize("placeholder", ["TODO", "tbd", "placeholder-paper", "paper_id_here"])
def test_normal_dataset_rejects_placeholder_looking_annotations(tmp_path, placeholder):
    path = tmp_path / "questions.jsonl"
    _write_jsonl(
        path,
        [
            {
                **_annotated(),
                "evidence_groups": [
                    {
                        "paper_id": placeholder,
                        "chunk_ids": [f"{placeholder}:p1:c0"],
                        "rationale": "Direct support.",
                    }
                ],
            }
        ],
    )
    with pytest.raises(DatasetError, match="^evaluation_dataset_unannotated$"):
        load_questions(path)


class FakeRetriever:
    def __init__(self, mode, rankings, events):
        self.mode = mode
        self.rankings = rankings
        self.events = events

    def search(self, question, *, k, paper_ids=None):
        self.events.append((self.mode, question, k))
        return [
            EvidenceChunk(
                chunk_id=chunk_id,
                paper_id=chunk_id.split(":", 1)[0],
                title="Stored title",
                page_number=1,
                text="canonical evidence",
            )
            for chunk_id in self.rankings[question]
        ]


class FakeChunkStore:
    def __init__(self, chunks):
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}

    def get_chunks_by_ids(self, chunk_ids):
        return [self.chunks[item] for item in chunk_ids if item in self.chunks]


class FakeQa:
    def __init__(self, retriever):
        self.retriever = retriever

    def answer(self, question):
        if question == "q1":
            return CitedAnswer(
                answer_markdown="not persisted",
                evidence_sufficient=True,
                citations=[
                    AnswerCitation(
                        chunk_id="p1:c2",
                        paper_id="untrusted-paper",
                        title="untrusted title",
                        page_number=999,
                        quote="untrusted raw output",
                    )
                ],
            )
        if self.retriever.mode == "vector":
            return CitedAnswer(
                answer_markdown="confident but unsupported",
                evidence_sufficient=True,
                citations=[],
            )
        return CitedAnswer(
            answer_markdown="insufficient",
            evidence_sufficient=False,
            citations=[],
        )


def _evaluation_fixture(tmp_path, *, hybrid=None):
    dataset = tmp_path / "questions.jsonl"
    _write_jsonl(
        dataset,
        [
            {
                "question_id": "q01",
                "question": "q1",
                "category": "exact_term",
                "evidence_groups": [
                    {
                        "paper_id": "p1",
                        "chunk_ids": ["p1:c2"],
                        "rationale": "The passage directly supports q1.",
                    }
                ],
            },
            {
                "question_id": "q02",
                "question": "q2",
                "category": "natural_language",
                "evidence_groups": [
                    {
                        "paper_id": "p2",
                        "chunk_ids": ["p2:c4"],
                        "rationale": "The passage directly supports q2.",
                    }
                ],
            },
        ],
    )
    events = []
    rankings = {
        "keyword": {"q1": ["p1:c2"], "q2": ["p2:c4"]},
        "vector": {"q1": ["other:c0", "p1:c2"], "q2": ["other:c0"]},
        "hybrid": hybrid or {"q1": ["p1:c2"], "q2": ["other:c0", "p2:c4"]},
        "two_stage": {"q1": ["p1:c2"], "q2": ["p2:c4"]},
    }
    retrievers = {
        mode: FakeRetriever(mode, ranking, events) for mode, ranking in rankings.items()
    }
    chunks = [
        EvidenceChunk(
            chunk_id=chunk_id,
            paper_id=paper_id,
            title="Canonical",
            page_number=1,
            text="canonical evidence",
        )
        for chunk_id, paper_id in (("p1:c2", "p1"), ("p2:c4", "p2"))
    ]
    return dataset, events, retrievers, FakeChunkStore(chunks)


def test_evaluator_uses_same_questions_and_calculates_exact_metrics(tmp_path):
    dataset, events, retrievers, store = _evaluation_fixture(tmp_path)

    result = run_evaluation(
        dataset,
        keyword_retriever=retrievers["keyword"],
        vector_retriever=retrievers["vector"],
        hybrid_retriever=retrievers["hybrid"],
        qa_factory=FakeQa,
        chunk_store=store,
        reports_dir=tmp_path / "reports",
        now=lambda: datetime(2026, 8, 13, 1, 2, 3, tzinfo=timezone.utc),
    )

    assert events == [
        (mode, question, 5)
        for mode in ("keyword", "vector", "hybrid")
        for question in ("q1", "q2")
    ]
    assert result.metrics["keyword"]["evidence_group_recall_at_5"] == 1.0
    assert result.metrics["keyword"]["mrr"] == 1.0
    assert result.metrics["vector"]["evidence_group_recall_at_5"] == 0.5
    assert result.metrics["vector"]["mrr"] == 0.25
    assert result.metrics["hybrid"]["evidence_group_recall_at_5"] == 1.0
    assert result.metrics["hybrid"]["mrr"] == 0.75
    assert result.metrics["overall"] == {
        "citation_precision": 1.0,
        "evidence_coverage": 0.5,
        "unsupported_claim_rate": 0.25,
    }

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert [item["question_id"] for item in payload["questions"]] == ["q01", "q02"]
    assert payload["acceptance"] == {"accepted": True, "code": "accepted"}
    assert set(payload) == {
        "acceptance",
        "config",
        "dataset_fingerprint",
        "dataset_summary",
        "metrics",
        "metrics_by_category",
        "questions",
        "timestamp_utc",
    }
    assert payload["dataset_summary"] == {
        "question_count": 2,
        "category_counts": {"exact_term": 1, "natural_language": 1},
        "evaluation_scope": "answer_and_retrieval",
    }
    assert payload["metrics_by_category"]["exact_term"]["keyword"]["evidence_group_recall_at_5"] == 1.0
    serialized = result.json_path.read_text(encoding="utf-8") + result.markdown_path.read_text(encoding="utf-8")
    for forbidden in ("canonical evidence", "untrusted raw output", "confident but unsupported", str(tmp_path)):
        assert forbidden not in serialized


def test_evaluator_includes_two_stage_and_reports_promotion_gate(tmp_path):
    dataset, events, retrievers, store = _evaluation_fixture(tmp_path)

    result = run_evaluation(
        dataset,
        keyword_retriever=retrievers["keyword"],
        vector_retriever=retrievers["vector"],
        hybrid_retriever=retrievers["hybrid"],
        two_stage_retriever=retrievers["two_stage"],
        qa_factory=FakeQa,
        chunk_store=store,
        reports_dir=tmp_path / "reports",
        include_answer_metrics=False,
    )

    assert events == [
        (mode, question, 5)
        for mode in ("keyword", "vector", "hybrid", "two_stage")
        for question in ("q1", "q2")
    ]
    assert set(result.metrics) == {"keyword", "vector", "hybrid", "two_stage", "overall"}
    assert result.acceptance["code"] == "two_stage_promotion_gate_failed"
    assert result.acceptance["checks"]["cross_paper_strict_improvement"] is False


def test_two_stage_promotion_gate_passes_when_cross_paper_improves_without_natural_language_regression():
    metrics = {
        "cross_paper": {
            "hybrid": {"paper_recall_at_5": 0.5, "evidence_group_recall_at_5": 0.5},
            "two_stage": {"paper_recall_at_5": 0.75, "evidence_group_recall_at_5": 0.5},
        },
        "natural_language": {
            "hybrid": {"evidence_group_recall_at_5": 0.8},
            "two_stage": {"evidence_group_recall_at_5": 0.8},
        },
    }

    acceptance = _two_stage_acceptance(metrics)

    assert acceptance["accepted"] is True
    assert acceptance["code"] == "two_stage_promotion_gate_passed"
    assert all(acceptance["checks"].values())


def test_two_stage_gate_fails_without_strict_cross_paper_improvement(tmp_path):
    dataset, _, retrievers, store = _evaluation_fixture(tmp_path)
    result = run_evaluation(
        dataset,
        keyword_retriever=retrievers["keyword"],
        vector_retriever=retrievers["vector"],
        hybrid_retriever=retrievers["hybrid"],
        two_stage_retriever=retrievers["hybrid"],
        qa_factory=FakeQa,
        chunk_store=store,
        reports_dir=tmp_path / "reports",
        include_answer_metrics=False,
    )

    assert result.acceptance["accepted"] is False
    assert result.acceptance["code"] == "two_stage_promotion_gate_failed"
    assert result.acceptance["checks"]["cross_paper_strict_improvement"] is False


def test_hybrid_recall_regression_fails_acceptance(tmp_path):
    dataset, _, retrievers, store = _evaluation_fixture(
        tmp_path, hybrid={"q1": ["other:c0"], "q2": ["other:c0"]}
    )
    result = run_evaluation(
        dataset,
        keyword_retriever=retrievers["keyword"],
        vector_retriever=retrievers["vector"],
        hybrid_retriever=retrievers["hybrid"],
        qa_factory=FakeQa,
        chunk_store=store,
        reports_dir=tmp_path / "reports",
    )
    assert result.acceptance == {
        "accepted": False,
        "code": "hybrid_recall_regression",
    }


def test_retrieval_only_evaluation_skips_qa_calls(tmp_path):
    dataset, events, retrievers, store = _evaluation_fixture(tmp_path)

    def fail_if_called(_retriever):
        raise AssertionError("qa_factory must not run in retrieval-only mode")

    result = run_evaluation(
        dataset,
        keyword_retriever=retrievers["keyword"],
        vector_retriever=retrievers["vector"],
        hybrid_retriever=retrievers["hybrid"],
        qa_factory=fail_if_called,
        chunk_store=store,
        reports_dir=tmp_path / "reports",
        include_answer_metrics=False,
    )

    assert result.metrics["hybrid"]["evidence_group_recall_at_5"] == 1.0
    assert result.metrics["overall"] == {
        "citation_precision": 0.0,
        "evidence_coverage": 0.0,
        "unsupported_claim_rate": 0.0,
    }
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["evaluation_scope"] == "retrieval_only"
    assert payload["dataset_summary"]["evaluation_scope"] == "retrieval_only"
    assert "仅代表当前本地语料版本" in result.markdown_path.read_text(encoding="utf-8")


def test_evaluation_report_persists_hybrid_retrieval_trace(tmp_path):
    dataset, _, retrievers, store = _evaluation_fixture(tmp_path)

    class TraceRetriever:
        def __init__(self, inner):
            self.inner = inner
            self.mode = inner.mode
            self.last_trace = RetrievalTrace(
                query="q", keyword_candidates=3, vector_candidates=4,
                fused_candidates=5, selected_count=2,
                selected_chunk_ids=["p1:c2"], selected_paper_ids=["p1"],
                latency_ms=1.5,
            )

        def search(self, question, *, k, paper_ids=None):
            result = self.inner.search(question, k=k, paper_ids=paper_ids)
            self.last_trace = self.last_trace.__class__(
                **{**self.last_trace.__dict__, "query": question}
            )
            return result

    hybrid = TraceRetriever(retrievers["hybrid"])

    class TraceQa(FakeQa):
        def __init__(self, retriever):
            super().__init__(retriever)
            self.last_trace = QATrace(
                status="answered", retrieval_ms=2.0, model_ms=8.0,
                retrieved_chunks=5, canonical_chunks=5, citation_count=2,
            )

    result = run_evaluation(
        dataset,
        keyword_retriever=retrievers["keyword"],
        vector_retriever=retrievers["vector"],
        hybrid_retriever=hybrid,
        qa_factory=TraceQa,
        chunk_store=store,
        reports_dir=tmp_path / "reports",
    )

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    trace = payload["questions"][0]["modes"]["hybrid"]["retrieval_trace"]
    assert trace["keyword_candidates"] == 3
    assert trace["selected_paper_ids"] == ["p1"]
    assert payload["questions"][0]["modes"]["hybrid"]["answer_trace"]["model_ms"] == 8.0


def test_report_names_are_collision_safe_and_failure_cleans_temps(tmp_path, monkeypatch):
    dataset, _, retrievers, store = _evaluation_fixture(tmp_path)
    kwargs = dict(
        keyword_retriever=retrievers["keyword"],
        vector_retriever=retrievers["vector"],
        hybrid_retriever=retrievers["hybrid"],
        qa_factory=FakeQa,
        chunk_store=store,
        reports_dir=tmp_path / "reports",
        now=lambda: datetime(2026, 8, 13, 1, 2, 3, tzinfo=timezone.utc),
    )
    first = run_evaluation(dataset, **kwargs)
    second = run_evaluation(dataset, **kwargs)
    assert first.json_path != second.json_path
    assert first.markdown_path != second.markdown_path

    monkeypatch.setattr("evaluation.run.os.replace", lambda *_: (_ for _ in ()).throw(OSError()))
    with pytest.raises(EvaluationError, match="^evaluation_report_write_failed$"):
        run_evaluation(dataset, **kwargs)
    assert list((tmp_path / "reports").glob("*.tmp")) == []


def test_provider_health_runs_four_isolated_probes_and_allows_only_safe_fields():
    events = []

    def ok():
        events.append("openalex")
        return {"status_code": 200, "remaining_quota": "42", "unsafe": "secret"}

    def fail():
        events.append("core")
        raise RuntimeError("https://host/?api_key=raw-secret")

    def missing():
        events.append("unpaywall")
        return {"status": "missing_configuration"}

    def http_error():
        events.append("crossref")
        return {"status_code": 429, "body": "private body"}

    result = run_provider_health(
        {
            "openalex": ok,
            "core": fail,
            "unpaywall": missing,
            "crossref": http_error,
        },
        clock=lambda: 1.0,
    )

    assert events == ["openalex", "core", "unpaywall", "crossref"]
    assert [item["provider"] for item in result] == [
        "openalex",
        "core",
        "unpaywall",
        "crossref",
    ]
    assert [item["status"] for item in result] == [
        "ok",
        "unreachable",
        "missing_configuration",
        "http_error",
    ]
    assert result[0]["remaining_quota"] == 42
    for item in result:
        assert set(item) <= {
            "provider",
            "status",
            "http_status",
            "latency_ms",
            "remaining_quota",
        }
    serialized = json.dumps(result)
    for forbidden in ("secret", "https://", "api_key", "private body"):
        assert forbidden not in serialized


def test_provider_health_cli_dispatches_injected_service_without_credentials():
    class Services:
        def provider_health(self):
            return [{"provider": "crossref", "status": "ok", "latency_ms": 1}]

    output = io.StringIO()
    assert app.run(["provider-health"], services=Services(), stdout=output) == 0
    assert json.loads(output.getvalue()) == [
        {"latency_ms": 1, "provider": "crossref", "status": "ok"}
    ]


def test_evaluate_cli_returns_stable_error_for_unannotated_template():
    class Services:
        def evaluate(self, dataset):
            load_questions(dataset)
            raise AssertionError("annotated dataset unexpectedly accepted")

    output = io.StringIO()
    exit_code = app.run(
        ["evaluate", "--dataset", "data/evaluation/questions.jsonl"],
        services=Services(),
        stdout=output,
    )
    assert exit_code != 0
    assert json.loads(output.getvalue()) == {
        "error": "evaluation_dataset_unannotated",
        "status": "error",
    }


def test_evaluate_cli_emits_injected_result_without_model_or_embedding():
    class Services:
        def evaluate(self, dataset):
            return {"status": "ok", "dataset": "safe-id"}

    output = io.StringIO()
    assert app.run(
        ["evaluate", "--dataset", "questions.jsonl"],
        services=Services(),
        stdout=output,
    ) == 0
    assert json.loads(output.getvalue()) == {"dataset": "safe-id", "status": "ok"}


def test_evaluate_cli_forwards_retrieval_only_flag():
    calls = []

    class Services:
        def evaluate(self, dataset, **kwargs):
            calls.append((dataset, kwargs))
            return {"status": "ok"}

    output = io.StringIO()
    assert app.run(
        ["evaluate", "--dataset", "questions.jsonl", "--retrieval-only"],
        services=Services(),
        stdout=output,
    ) == 0
    assert calls == [("questions.jsonl", {"retrieval_only": True})]


def test_readme_is_truthful_and_contains_required_boundaries():
    readme = Path("README.md").read_text(encoding="utf-8")
    required = (
        "水色遥感预测科研助手",
        "OpenAlex",
        "Unpaywall",
        "CORE",
        "Crossref",
        "不绕过付费墙",
        "受控 Agent",
        "失败可解释",
        "合法开放全文",
        "streamlit run web/app.py",
        "索引重建",
        "证据和安全边界",
        "评估口径",
        "已知限制",
        "questions-annotated.jsonl",
    )
    assert all(item in readme for item in required)
    for outdated in ("arXiv API", "Tavily", "Bing"):
        assert outdated not in readme
    assert "sk-" not in readme


def test_answer_level_metrics_require_supported_citations_and_count_fallbacks():
    rows = [
        {
            "question_id": "q1",
            "relevant_chunk_ids": ["p1:c1"],
            "claims": [{"text": "supported", "evidence_sufficient": True, "citation_chunk_ids": ["p1:c1"]}],
        },
        {
            "question_id": "q2",
            "relevant_chunk_ids": ["p2:c1"],
            "claims": [{"text": "unsupported", "evidence_sufficient": True, "citation_chunk_ids": ["p9:c1"]}],
        },
        {
            "question_id": "q3",
            "relevant_chunk_ids": ["p3:c1"],
            "claims": [{"text": "fallback", "evidence_sufficient": False, "citation_chunk_ids": []}],
        },
    ]

    assert answer_level_metrics(rows) == {
        "citation_precision": 0.5,
        "evidence_coverage": 1 / 3,
        "unsupported_claim_rate": 0.5,
        "question_count": 3,
    }


def test_answer_dataset_loader_rejects_sensitive_content(tmp_path):
    path = tmp_path / "answers.jsonl"
    path.write_text(
        json.dumps({
            "question_id": "q1",
            "relevant_chunk_ids": ["p1:c1"],
            "claims": [{
                "text": "https://not-allowed.example",
                "evidence_sufficient": True,
                "citation_chunk_ids": ["p1:c1"],
            }],
        }) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="^evaluation_answer_dataset_sensitive_content$"):
        load_answer_rows(path)


def test_evaluation_rejects_report_config_that_could_leak_credentials(tmp_path):
    dataset, _, retrievers, store = _evaluation_fixture(tmp_path)
    with pytest.raises(EvaluationError, match="^evaluation_config_invalid$"):
        run_evaluation(
            dataset,
            keyword_retriever=retrievers["keyword"],
            vector_retriever=retrievers["vector"],
            hybrid_retriever=retrievers["hybrid"],
            qa_factory=FakeQa,
            chunk_store=store,
            reports_dir=tmp_path / "reports",
            config={"api_key": "must-not-appear"},
        )
    assert not (tmp_path / "reports").exists()


@pytest.mark.parametrize(
    "config",
    [
        {"retrieval_k": 5.5, "hybrid": {"keyword_weight": 1.0, "vector_weight": 1.0, "rrf_k": 60}},
        {"retrieval_k": 5, "hybrid": {"keyword_weight": math.nan, "vector_weight": 1.0, "rrf_k": 60}},
        {"retrieval_k": 5, "hybrid": {"keyword_weight": 1.0, "vector_weight": 1.0, "rrf_k": 60.5}},
    ],
)
def test_evaluation_rejects_fractional_or_nonfinite_config(tmp_path, config):
    dataset, _, retrievers, store = _evaluation_fixture(tmp_path)
    with pytest.raises(EvaluationError, match="^evaluation_config_invalid$"):
        run_evaluation(
            dataset,
            keyword_retriever=retrievers["keyword"],
            vector_retriever=retrievers["vector"],
            hybrid_retriever=retrievers["hybrid"],
            qa_factory=FakeQa,
            chunk_store=store,
            reports_dir=tmp_path / "reports",
            config=config,
        )


def test_evaluation_rejects_non_five_retrieval_k_to_keep_recall_label_truthful(tmp_path):
    dataset, _, retrievers, store = _evaluation_fixture(tmp_path)
    with pytest.raises(EvaluationError, match="^evaluation_config_invalid$"):
        run_evaluation(
            dataset,
            keyword_retriever=retrievers["keyword"],
            vector_retriever=retrievers["vector"],
            hybrid_retriever=retrievers["hybrid"],
            qa_factory=FakeQa,
            chunk_store=store,
            reports_dir=tmp_path / "reports",
            config={
                "retrieval_k": 3,
                "hybrid": {"keyword_weight": 1.0, "vector_weight": 1.0, "rrf_k": 60},
            },
        )


def test_evaluator_composes_hybrid_with_reported_weights(tmp_path):
    dataset, events, retrievers, store = _evaluation_fixture(tmp_path)
    config = {
        "retrieval_k": 5,
        "hybrid": {"keyword_weight": 2.0, "vector_weight": 0.5, "rrf_k": 7},
    }
    result = run_evaluation(
        dataset,
        keyword_retriever=retrievers["keyword"],
        vector_retriever=retrievers["vector"],
        hybrid_factory=lambda keyword, vector, **kwargs: FakeRetriever(
            "hybrid-measured", {"q1": ["p1:c2"], "q2": ["p2:c4"]}, events
        ) if kwargs == {
            "keyword_weight": 2.0,
            "vector_weight": 0.5,
            "rrf_k": 7,
        } else (_ for _ in ()).throw(AssertionError("reported weights not applied")),
        qa_factory=FakeQa,
        chunk_store=store,
        reports_dir=tmp_path / "reports",
        config=config,
    )
    assert events == [
        (mode, question, 5)
        for mode in ("keyword", "vector", "hybrid-measured")
        for question in ("q1", "q2")
    ]
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["config"] == config


def test_default_evaluation_uses_runtime_hybrid_weights(tmp_path, monkeypatch):
    dataset, _, _, _ = _evaluation_fixture(tmp_path)
    captured = {}
    database = object()
    services = app.DefaultServices()

    monkeypatch.setattr(services, "_database", lambda: database)
    monkeypatch.setattr(services, "_vector_store", lambda value: object())
    monkeypatch.setattr(services, "_model", lambda: object())
    monkeypatch.setattr("retrieval.keyword_index.KeywordIndex", lambda value: object())
    monkeypatch.setattr(
        "utils.config.load_rag_config",
        lambda: {
            "keyword_weight": 20.0,
            "vector_weight": 1.0,
            "rrf_k": 60,
        },
    )

    def fake_run_evaluation(*args, **kwargs):
        captured.update(kwargs)
        return "measured"

    monkeypatch.setattr("evaluation.run.run_evaluation", fake_run_evaluation)

    assert services.evaluate(dataset, retrieval_only=True) == "measured"
    assert captured["config"] == {
        "retrieval_k": 5,
        "hybrid": {
            "keyword_weight": 20.0,
            "vector_weight": 1.0,
            "rrf_k": 60,
        },
        "two_stage": {
            "paper_candidate_k": 50,
            "paper_k": 12,
            "chunk_candidate_k": 40,
            "max_chunks_per_paper": 2,
        },
    }


def test_default_cited_qa_composition_keeps_hybrid_retriever_available(monkeypatch):
    services = app.DefaultServices()
    monkeypatch.setattr(services, "_database", lambda: object())
    monkeypatch.setattr(services, "_vector_store", lambda database: object())
    monkeypatch.setattr(services, "_model", lambda: object())
    monkeypatch.setattr("workflows.qa.CitedQaService.answer", lambda self, question: "safe")

    assert services.cited_qa("question") == "safe"


def test_successful_evaluation_result_is_cli_json_serializable(tmp_path):
    dataset, _, retrievers, store = _evaluation_fixture(tmp_path)

    class Services:
        def evaluate(self, dataset_path):
            return run_evaluation(
                dataset_path,
                keyword_retriever=retrievers["keyword"],
                vector_retriever=retrievers["vector"],
                hybrid_retriever=retrievers["hybrid"],
                qa_factory=FakeQa,
                chunk_store=store,
                reports_dir=tmp_path / "reports",
            )

    output = io.StringIO()
    assert app.run(
        ["evaluate", "--dataset", str(dataset)], services=Services(), stdout=output
    ) == 0
    payload = json.loads(output.getvalue())
    assert payload["status"] == "ok"
    assert payload["acceptance"]["code"] == "accepted"
    assert payload["json_report"].endswith(".json")
    assert str(tmp_path) not in output.getvalue()


def test_default_provider_health_uses_exactly_one_bounded_request_per_configured_provider(monkeypatch):
    from providers.health import default_provider_health

    calls = []

    class Response:
        status_code = 200
        headers = {"X-RateLimit-Remaining": "7"}

    def requester(url, *, headers, params, timeout, allow_redirects):
        calls.append((url, timeout, allow_redirects))
        return Response()

    result = default_provider_health(
        request_get=requester,
        environ={
            "OPENALEX_API_KEY": "configured",
            "CORE_API_KEY": "configured",
            "UNPAYWALL_EMAIL": "configured",
        },
    )

    assert len(calls) == 4
    assert all(timeout == 10 and allow_redirects is False for _, timeout, allow_redirects in calls)
    assert [item["status"] for item in result] == ["ok", "ok", "ok", "ok"]


def test_default_provider_health_probes_configured_semantic_scholar_once(monkeypatch):
    from providers.health import default_provider_health

    calls = []

    class Response:
        status_code = 200
        headers = {"X-RateLimit-Remaining": "4"}

    def requester(url, *, headers, params, timeout, allow_redirects):
        calls.append((url, headers, params, timeout, allow_redirects))
        return Response()

    result = default_provider_health(
        request_get=requester,
        environ={"SEMANTIC_SCHOLAR_API_KEY": "configured"},
    )

    assert len(calls) == 3
    assert len(result) == 5
    assert result[-1]["provider"] == "semantic_scholar"
    assert result[-1]["status"] == "ok"
    assert calls[-1][1] == {"x-api-key": "configured"}
    assert calls[-1][4] is False


def test_provider_http_exception_is_sanitized_as_http_error():
    class Response:
        status_code = 503

    class HttpFailure(RuntimeError):
        response = Response()

    result = run_provider_health(
        {
            "openalex": lambda: (_ for _ in ()).throw(HttpFailure("secret URL")),
            "core": lambda: {"status": "missing_configuration"},
            "unpaywall": lambda: {"status": "missing_configuration"},
            "crossref": lambda: {"status_code": 200},
        }
    )
    assert result[0]["status"] == "http_error"
    assert result[0]["http_status"] == 503
    assert "secret" not in json.dumps(result)


def test_default_provider_health_does_not_retry_http_failures(monkeypatch):
    from providers.health import default_provider_health

    calls = []

    class Response:
        status_code = 503
        headers = {}

    def single_request(*args, **kwargs):
        calls.append(args[0])
        return Response()

    monkeypatch.setattr("requests.get", single_request)
    result = default_provider_health(
        environ={
            "OPENALEX_API_KEY": "configured",
            "CORE_API_KEY": "configured",
            "UNPAYWALL_EMAIL": "configured",
        }
    )

    assert len(calls) == 4
    assert [item["status"] for item in result] == [
        "http_error",
        "http_error",
        "http_error",
        "http_error",
    ]


def test_provider_health_selected_path_loads_dotenv_lazily_and_disables_redirects(monkeypatch):
    import providers.health as health

    loaded = []
    calls = []

    class Response:
        status_code = 200
        headers = {}

    def loader():
        loaded.append("dotenv")

    def requester(url, *, headers, params, timeout, allow_redirects):
        calls.append(allow_redirects)
        return Response()

    result = health.default_provider_health(
        request_get=requester,
        environ={
            "OPENALEX_API_KEY": "configured",
            "CORE_API_KEY": "configured",
            "UNPAYWALL_EMAIL": "configured",
        },
        dotenv_loader=loader,
    )

    assert loaded == ["dotenv"]
    assert calls == [False, False, False, False]
    assert all(item["status"] == "ok" for item in result)
