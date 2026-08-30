from __future__ import annotations

from collections import Counter
from datetime import datetime
import io
import json
from pathlib import Path

import pytest
import yaml

from domain.models import IngestionResult, PaperCandidate
from domain.statuses import ABSTRACT_ONLY, DISCOVERED, FAILED, INDEXED, PARSED, PROFILED
from storage.database import ResearchDatabase
from workflows.seed import SeedService, _usable_oa
import app


ROOT = Path(__file__).resolve().parents[1]


def paper(
    number: int,
    *,
    year: int,
    citations: int,
    oa: bool = False,
    doi: str | None = None,
    title: str | None = None,
) -> PaperCandidate:
    return PaperCandidate(
        source="openalex",
        source_id=f"W{number}",
        title=title or f"Water prediction paper {number}",
        doi=doi if doi is not None else f"10.1000/{number}",
        year=year,
        cited_by_count=citations,
        abstract=f"Abstract {number}",
        pdf_url=f"https://oa.example/{number}.pdf" if oa else None,
        license="cc-by" if oa else None,
    )


class FakeRegistry:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query
        self.calls = []

    def discover(self, query, *, from_year, max_results):
        self.calls.append((query, from_year, max_results))
        return list(self.results_by_query.get(query, ()))[:max_results]


class TrackingDatabase:
    def __init__(self, path, events):
        self.inner = ResearchDatabase(path)
        self.events = events

    def upsert_candidate(self, candidate):
        self.events.append(("persist", candidate.source_id))
        return self.inner.upsert_candidate(candidate)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class FakeIngestor:
    def __init__(self, database, events, *, fail_once=None, terminal=INDEXED):
        self.database = database
        self.events = events
        self.fail_once = set(fail_once or ())
        self.failed = set()
        self.terminal = terminal

    def ingest(self, candidate):
        self.events.append(("ingest", candidate.source_id))
        if candidate.source_id in self.fail_once and candidate.source_id not in self.failed:
            self.failed.add(candidate.source_id)
            raise RuntimeError("secret-token-must-not-escape")
        record = self.database.find_candidate(candidate)
        self.database.update_status(record.paper_id, self.terminal)
        return IngestionResult(paper_id=record.paper_id, status=self.terminal)


def config(*, metadata=8, fulltext=4, recent=3, representative=2):
    return {
        "domain": "water-color remote-sensing prediction",
        "from_year": 2019,
        "target_metadata": metadata,
        "target_fulltext": fulltext,
        "queries": ["q1", "q2"],
        "recent_queries": {"from_year": 2024, "target": recent},
        "representative_queries": {"sort": "cited_by_count", "target": representative},
    }


def test_exact_seed_configuration():
    loaded = yaml.safe_load((ROOT / "config" / "seed_queries.yml").read_text(encoding="utf-8"))
    assert loaded == {
        "domain": "water-color remote-sensing prediction",
        "from_year": 2019,
        "target_metadata": 480,
        "target_fulltext": 160,
        "queries": [
            "water color remote sensing prediction",
            "chlorophyll-a remote sensing machine learning",
            "turbidity Secchi depth remote sensing prediction",
            "harmful algal bloom remote sensing forecasting",
            "Sentinel-2 water quality parameter estimation",
            "Landsat inland water quality prediction",
            "deep learning inland water remote sensing",
        ],
        "recent_queries": {"from_year": 2024, "target": 100},
        "representative_queries": {"sort": "cited_by_count", "target": 60},
    }


def test_seed_configuration_targets_bounded_personal_corpus():
    loaded = yaml.safe_load((ROOT / "config" / "seed_queries.yml").read_text(encoding="utf-8"))

    assert 450 <= loaded["target_metadata"] <= 500
    assert loaded["target_fulltext"] <= loaded["target_metadata"]
    assert len(loaded["queries"]) == 7
    assert loaded["recent_queries"]["target"] + loaded["representative_queries"]["target"] < loaded["target_metadata"]


def test_crossref_doi_is_deferred_to_unpaywall_for_fulltext_resolution():
    assert _usable_oa(paper(1, year=2024, citations=1, oa=False, title="Crossref", doi="10.1000/crossref" ).model_copy(update={"source": "crossref"}))


def test_selection_deduplicates_reserves_recent_and_representative_and_caps_targets(tmp_path):
    candidates = [
        paper(1, year=2019, citations=500),
        paper(2, year=2020, citations=400),
        paper(3, year=2024, citations=2, oa=True, doi=""),
        paper(4, year=2025, citations=1, oa=True),
        paper(5, year=2026, citations=0),
        paper(6, year=2021, citations=40, oa=True),
        paper(7, year=2022, citations=30),
        paper(8, year=2023, citations=20, oa=True),
        paper(9, year=2024, citations=10),
        paper(10, year=2025, citations=8, oa=True),
        paper(11, year=2026, citations=7),
        paper(12, year=2020, citations=6, oa=True),
    ]
    doi_duplicate = paper(101, year=2019, citations=999, doi="HTTPS://DOI.ORG/10.1000/1")
    title_duplicate = paper(102, year=2024, citations=999, oa=True, doi="", title="Water prediction paper 3!!!")
    registry = FakeRegistry({"q1": candidates[:6] + [doi_duplicate], "q2": [title_duplicate] + candidates[6:]})
    events = []
    database = TrackingDatabase(tmp_path / "research.db", events)
    ingestor = FakeIngestor(database, events)

    report = SeedService(registry, ingestor, database).collect(config())

    stored = database.list_papers(limit=100)
    assert report.metadata_count == 8
    assert len(stored) == 8
    assert {p.source_id for p in stored} >= {"W101", "W2", "W4", "W102"}
    assert sum(p.normalized_title == "water prediction paper 3" for p in stored) == 1
    assert len([event for event in events if event[0] == "ingest"]) == 4
    assert all(p.pdf_url and p.license for p in stored if p.status == INDEXED)
    assert all(call[1:] == (2019, 8) for call in registry.calls)


def test_equal_score_prefers_usable_oa_and_order_is_deterministic(tmp_path):
    closed = paper(1, year=2023, citations=10)
    opened = paper(2, year=2023, citations=10, oa=True)
    # Both are first-ranked in separate queries, so all non-OA score components tie.
    registry = FakeRegistry({"q1": [closed], "q2": [opened]})
    events = []
    database = TrackingDatabase(tmp_path / "research.db", events)
    ingestor = FakeIngestor(database, events)

    report = SeedService(registry, ingestor, database).collect(
        config(metadata=1, fulltext=1, recent=0, representative=0)
    )

    assert report.metadata_count == 1
    assert database.list_papers(limit=10)[0].source_id == "W2"
    assert events == [("persist", "W2"), ("ingest", "W2")]


@pytest.mark.parametrize(
    "change, code",
    [
        ({"target_metadata": 0}, "seed_invalid_target_metadata"),
        ({"target_fulltext": 0}, "seed_invalid_target_fulltext"),
        ({"target_fulltext": -1}, "seed_invalid_target_fulltext"),
        ({"target_metadata": 1, "target_fulltext": 2}, "seed_fulltext_exceeds_metadata"),
        ({"queries": []}, "seed_queries_required"),
    ],
)
def test_invalid_seed_configuration_has_stable_errors(tmp_path, change, code):
    cfg = config()
    cfg.update(change)
    service = SeedService(FakeRegistry({}), FakeIngestor(None, []), ResearchDatabase(tmp_path / "db.sqlite"))
    with pytest.raises(ValueError, match=f"^{code}$"):
        service.collect(cfg)


def test_persists_all_selected_before_ingestion_and_resumes_terminal_fingerprints(tmp_path):
    candidates = [paper(i, year=2024, citations=20 - i, oa=True) for i in range(1, 5)]
    events = []
    database = TrackingDatabase(tmp_path / "research.db", events)
    registry = FakeRegistry({"q1": candidates, "q2": []})
    first = FakeIngestor(database, events, fail_once={"W2"})
    service = SeedService(registry, first, database)

    first_report = service.collect(config(metadata=4, fulltext=4, recent=0, representative=0))

    first_ingest = next(i for i, event in enumerate(events) if event[0] == "ingest")
    assert all(event[0] == "persist" for event in events[:first_ingest])
    assert first_report.failures == {"ingestion_failed": 1}
    assert "secret-token" not in first_report.model_dump_json()

    events.clear()
    second = FakeIngestor(database, events)
    second_report = SeedService(registry, second, database).collect(
        config(metadata=4, fulltext=4, recent=0, representative=0)
    )

    assert [event for event in events if event[0] == "ingest"] == [("ingest", "W2")]
    assert second_report.indexed_count == 1
    assert second_report.failures == {}


def test_unchanged_abstract_only_is_terminal_but_changed_fingerprint_retries(tmp_path):
    candidate = paper(1, year=2024, citations=1, oa=True)
    events = []
    database = TrackingDatabase(tmp_path / "research.db", events)
    stored = database.upsert_candidate(candidate)
    database.update_status(stored.paper_id, ABSTRACT_ONLY)
    registry = FakeRegistry({"q1": [candidate], "q2": []})
    ingestor = FakeIngestor(database, events, terminal=ABSTRACT_ONLY)

    SeedService(registry, ingestor, database).collect(config(metadata=1, fulltext=1, recent=0, representative=0))
    assert ("ingest", "W1") not in events

    changed = candidate.model_copy(update={"source_updated_at": "2026-08-13"})
    registry.results_by_query["q1"] = [changed]
    events.clear()
    SeedService(registry, ingestor, database).collect(config(metadata=1, fulltext=1, recent=0, representative=0))
    assert events == [("persist", "W1"), ("ingest", "W1")]


def test_offline_parsed_papers_are_terminal_until_model_services_are_available(tmp_path):
    candidate_item = paper(1, year=2024, citations=1, oa=True)
    events = []
    database = TrackingDatabase(tmp_path / "research.db", events)
    registry = FakeRegistry({"q1": [candidate_item], "q2": []})
    ingestor = FakeIngestor(database, events, terminal=PARSED)
    ingestor.offline_only = True
    service = SeedService(registry, ingestor, database)

    service.collect(config(metadata=1, fulltext=1, recent=0, representative=0))
    events.clear()
    service.collect(config(metadata=1, fulltext=1, recent=0, representative=0))

    assert [event for event in events if event[0] == "ingest"] == []


def test_changed_terminal_record_is_reset_before_real_ingestor_idempotency_check(tmp_path):
    candidate = paper(1, year=2024, citations=1, oa=True)
    database = ResearchDatabase(tmp_path / "research.db")
    stored = database.upsert_candidate(candidate)
    database.update_status(stored.paper_id, INDEXED)
    changed = candidate.model_copy(update={"source_updated_at": "2026-08-13"})
    statuses_seen = []

    class IdempotencyProbe:
        def ingest(self, item):
            current = database.find_candidate(item)
            statuses_seen.append(current.status)
            return IngestionResult(paper_id=current.paper_id, status=current.status, skipped=True)

    SeedService(FakeRegistry({"q1": [changed], "q2": []}), IdempotencyProbe(), database).collect(
        config(metadata=1, fulltext=1, recent=0, representative=0)
    )
    assert statuses_seen == [DISCOVERED]


def test_dedup_keeps_distinct_dois_even_when_title_and_year_match(tmp_path):
    first = paper(1, year=2024, citations=10, title="Same title")
    second = paper(2, year=2024, citations=9, title="Same title")
    database = ResearchDatabase(tmp_path / "research.db")
    report = SeedService(
        FakeRegistry({"q1": [first, second], "q2": []}),
        FakeIngestor(database, []),
        database,
    ).collect(config(metadata=2, fulltext=1, recent=0, representative=0))
    assert report.metadata_count == 2
    assert database.count_papers() == 2


@pytest.mark.parametrize("doi_first", [False, True])
def test_dedup_merges_doi_and_no_doi_title_fallback_in_both_orders(tmp_path, doi_first):
    with_doi = paper(1, year=2024, citations=10, title="Fallback identity")
    without_doi = paper(2, year=2024, citations=9, title="Fallback identity", doi="")
    ordered = [with_doi, without_doi] if doi_first else [without_doi, with_doi]
    database = ResearchDatabase(tmp_path / "research.db")
    report = SeedService(
        FakeRegistry({"q1": ordered, "q2": []}), FakeIngestor(database, []), database
    ).collect(config(metadata=2, fulltext=1, recent=0, representative=0))
    assert report.metadata_count == 1
    assert database.count_papers() == 1


def test_failure_counts_are_stably_sorted(tmp_path):
    candidates = [paper(i, year=2025, citations=i, oa=True) for i in range(1, 4)]
    events = []
    database = TrackingDatabase(tmp_path / "research.db", events)

    class FailingIngestor:
        def ingest(self, candidate):
            if candidate.source_id == "W1":
                raise RuntimeError("credential=should-not-leak")
            record = database.find_candidate(candidate)
            database.update_status(record.paper_id, FAILED, "pdf_parse_failed")
            return IngestionResult(paper_id=record.paper_id, status=FAILED)

    report = SeedService(FakeRegistry({"q1": candidates, "q2": []}), FailingIngestor(), database).collect(
        config(metadata=3, fulltext=3, recent=0, representative=0)
    )
    assert report.failures == {"ingestion_failed": 1, "pdf_parse_failed": 2}
    assert "credential" not in report.model_dump_json()


def test_abstract_only_fulltext_attempt_counts_stable_reason(tmp_path):
    candidate = paper(1, year=2025, citations=1, oa=True)
    database = ResearchDatabase(tmp_path / "research.db")

    class AbstractOnlyIngestor:
        def ingest(self, item):
            record = database.find_candidate(item)
            database.update_status(record.paper_id, ABSTRACT_ONLY, "invalid_pdf")
            return IngestionResult(paper_id=record.paper_id, status=ABSTRACT_ONLY)

    report = SeedService(
        FakeRegistry({"q1": [candidate], "q2": []}), AbstractOnlyIngestor(), database
    ).collect(config(metadata=1, fulltext=1, recent=0, representative=0))
    assert report.failures == {"invalid_pdf": 1}


def test_unknown_database_error_is_mapped_to_stable_failure_code(tmp_path):
    candidate = paper(1, year=2025, citations=1, oa=True)
    database = ResearchDatabase(tmp_path / "research.db")

    class UnsafeFailureIngestor:
        def ingest(self, item):
            record = database.find_candidate(item)
            database.update_status(
                record.paper_id, FAILED, "credential=raw-secret https://unsafe.example/?token=x"
            )
            return IngestionResult(paper_id=record.paper_id, status=FAILED)

    report = SeedService(
        FakeRegistry({"q1": [candidate], "q2": []}), UnsafeFailureIngestor(), database
    ).collect(config(metadata=1, fulltext=1, recent=0, representative=0))
    assert report.failures == {"ingestion_failed": 1}
    assert "secret" not in report.model_dump_json()


def test_overlap_counts_toward_both_reservations_before_score_fill(tmp_path):
    overlap = paper(1, year=2025, citations=100)
    representative_only = paper(2, year=2020, citations=90)
    recent_best = paper(3, year=2026, citations=10)
    recent_extra = paper(4, year=2024, citations=1)
    score_fill = paper(5, year=2023, citations=80, oa=True)
    database = ResearchDatabase(tmp_path / "research.db")
    SeedService(
        FakeRegistry({"q1": [overlap, representative_only, score_fill, recent_best, recent_extra], "q2": []}),
        FakeIngestor(database, []),
        database,
    ).collect(config(metadata=4, fulltext=1, recent=2, representative=2))
    selected = {item.source_id for item in database.list_papers(limit=10)}
    assert selected == {"W1", "W2", "W3", "W5"}


def test_disjoint_reservations_both_survive_when_combined_targets_exceed_cap(tmp_path):
    representative = paper(1, year=2020, citations=100)
    representative_two = paper(2, year=2021, citations=90)
    recent = paper(3, year=2026, citations=2)
    recent_two = paper(4, year=2025, citations=1)
    database = ResearchDatabase(tmp_path / "research.db")
    SeedService(
        FakeRegistry({"q1": [representative, representative_two], "q2": [recent, recent_two]}),
        FakeIngestor(database, []),
        database,
    ).collect(config(metadata=2, fulltext=1, recent=2, representative=2))
    assert {item.source_id for item in database.list_papers(limit=10)} == {"W1", "W3"}


def test_recency_zero_span_current_year_and_missing_year_are_defined(tmp_path):
    current_year = datetime.now().year
    ranked = SeedService._rank(
        [
            {"candidate": paper(1, year=current_year, citations=0), "relevance": 1.0},
            {"candidate": paper(2, year=current_year, citations=0).model_copy(update={"year": None}), "relevance": 1.0},
        ],
        current_year,
    )
    assert ranked[0].recency == 1.0
    assert ranked[1].recency == 0.0


@pytest.mark.parametrize(
    "argv, command",
    [
        (["seed", "--config", "config/seed_queries.yml"], "seed"),
        (["retry-fulltext", "--limit", "2"], "retry-fulltext"),
        (["sync"], "sync"),
        (["stats"], "stats"),
        (["ask", "Which models are used for chlorophyll-a prediction?"], "ask"),
        (["rebuild-index"], "rebuild-index"),
        (["evaluate", "--dataset", "data/evaluation/questions.jsonl"], "evaluate"),
        (["evaluate-answers", "--dataset", "data/evaluation/answers-annotated.jsonl"], "evaluate-answers"),
    ],
)
def test_parser_recognizes_exact_commands_without_building_dependencies(argv, command):
    parser = app.build_parser()
    args = parser.parse_args(argv)
    assert args.command == command


def test_each_command_dispatches_to_only_its_injected_service(tmp_path):
    calls = []

    class Services:
        def seed(self, config):
            calls.append(("seed", config["target_metadata"]))
            return {"status": "ok"}

        def retry_fulltext(self, limit):
            calls.append(("retry_fulltext", limit))
            return {"status": "ok"}

        def sync(self):
            calls.append(("sync",))
            return {"status": "ok"}

        def stats(self):
            calls.append(("stats",))
            return {"metadata_total": 0}

        def cited_qa(self, question):
            calls.append(("cited_qa", question))
            return {"answer_markdown": "grounded"}

        def rebuild_index(self):
            calls.append(("rebuild",))
            return 3

        def evaluate(self, dataset):
            calls.append(("evaluate", str(dataset)))
            return {"status": "ok"}

        def evaluate_answers(self, dataset):
            calls.append(("evaluate_answers", str(dataset)))
            return {"status": "ok"}

    cfg_path = tmp_path / "seed.yml"
    cfg_path.write_text("target_metadata: 7\n", encoding="utf-8")
    scenarios = [
        (["seed", "--config", str(cfg_path)], "seed"),
        (["retry-fulltext", "--limit", "2"], "retry_fulltext"),
        (["sync"], "sync"),
        (["stats"], "stats"),
        (["ask", "question"], "cited_qa"),
        (["rebuild-index"], "rebuild"),
        (["evaluate", "--dataset", "questions.jsonl"], "evaluate"),
        (["evaluate-answers", "--dataset", "answers.jsonl"], "evaluate_answers"),
    ]
    for argv, expected in scenarios:
        calls.clear()
        output = io.StringIO()
        assert app.run(argv, services=Services(), stdout=output) == 0
        assert len(calls) == 1 and calls[0][0] == expected
        assert "OPENAI_API_KEY" not in output.getvalue()


def test_retry_fulltext_forwards_provider_rate_and_timeout():
    calls = []

    class Services:
        def retry_fulltext(self, limit, **kwargs):
            calls.append((limit, kwargs))
            return {"status": "ok"}

    output = io.StringIO()
    assert app.run(
        [
            "retry-fulltext",
            "--limit", "2",
            "--provider", "semantic_scholar",
            "--request-rate", "0.8",
            "--download-timeout", "3",
        ],
        services=Services(),
        stdout=output,
    ) == 0
    assert calls == [(2, {
        "discovered_only": False,
        "provider": "semantic_scholar",
        "request_rate": 0.8,
        "download_timeout": 3,
    })]


def test_sync_wrapper_returns_stable_counts(monkeypatch):
    from utils import pipeline

    monkeypatch.setattr(pipeline, "collect_papers_and_ingest", lambda: "collected=7 ingested=5")
    assert pipeline.sync_papers() == {"status": "ok", "collected": 7, "ingested": 5}


def test_sync_wrapper_sanitizes_operation_exception(monkeypatch):
    from utils import pipeline

    def fail():
        raise RuntimeError("credential=raw-secret")

    monkeypatch.setattr(pipeline, "collect_papers_and_ingest", fail)
    result = pipeline.sync_papers()
    assert result == {"status": "sync_failed", "collected": 0, "ingested": 0}
    assert "secret" not in json.dumps(result)


def test_stats_reports_only_counts_and_parameterized_provider_aggregation(tmp_path):
    database = ResearchDatabase(tmp_path / "research.db")
    first = database.upsert_candidate(paper(1, year=2024, citations=1, oa=True))
    second = database.upsert_candidate(paper(2, year=2024, citations=1).model_copy(update={"source": "core"}))
    database.update_status(first.paper_id, INDEXED)
    database.update_status(second.paper_id, FAILED, "secret=https://example.test/?key=credential")

    stats = database.catalog_statistics(
        status_names=("pdf_ready", "parsed", INDEXED, ABSTRACT_ONLY, FAILED)
    )

    assert stats == {
        "metadata_total": 2,
        "pdf_ready": 0,
        "parsed": 0,
        "indexed": 1,
        "abstract_only": 0,
        "failed": 1,
        "providers": {"core": 1, "openalex": 1},
    }
    assert "http" not in json.dumps(stats)


def test_default_stats_includes_profiled_status(monkeypatch):
    captured = {}

    class FakeDatabase:
        def catalog_statistics(self, *, status_names):
            captured["status_names"] = status_names
            return {"metadata_total": 2, "profiled": 2}

    monkeypatch.setattr(app.DefaultServices, "_database", staticmethod(lambda: FakeDatabase()))

    stats = app.DefaultServices().stats()

    assert PROFILED in captured["status_names"]
    assert stats["profiled"] == 2


def test_default_rebuild_skips_loading_incompatible_active_index(monkeypatch):
    captured = {}

    class FakeVectorStore:
        def rebuild_from_database(self):
            return 7

    services = app.DefaultServices()
    monkeypatch.setattr(services, "_database", lambda: object())

    def build_vector_store(database, *, load_existing=True):
        captured["database"] = database
        captured["load_existing"] = load_existing
        return FakeVectorStore()

    monkeypatch.setattr(services, "_vector_store", build_vector_store)

    assert services.rebuild_index() == 7
    assert captured["load_existing"] is False


def test_evaluate_validates_template_before_building_model_dependencies():
    output = io.StringIO()
    exit_code = app.run(
        ["evaluate", "--dataset", "data/evaluation/questions.jsonl"],
        services=app.DefaultServices(),
        stdout=output,
    )
    assert exit_code != 0
    assert json.loads(output.getvalue()) == {
        "error": "evaluation_dataset_unannotated",
        "status": "error",
    }
