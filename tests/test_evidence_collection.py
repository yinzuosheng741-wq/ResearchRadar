"""Dataset files cannot hide split leakage or duplicate tasks from one another."""

import json
import sys
from pathlib import Path

import pytest

from evaluation.evidence_cases import (
    AnnotationError,
    audit_case_collection,
    cases_hash,
    load_cases,
)
from tests.test_evidence_cases import canonical, case, frozen, write_rows


def pair(tmp_path):
    first, second = case(), case()
    second.update(case_id="pilot-2", origin_draft_id="draft-2", question="Another question about a different study?",
                  split="holdout", paper_scope=["paper-b"], source_groups=["10.1234/paper-b"])
    second["evidence"][0].update(chunk_id="chunk-b", paper_id="paper-b", source_group="10.1234/paper-b")
    return first, second


@pytest.mark.parametrize("mode", ["doi", "paper", "template", "question", "case_id"])
def test_cross_file_leakage_is_rejected_even_when_each_file_is_valid(tmp_path, mode):
    first, second = pair(tmp_path)
    if mode == "doi":
        second["source_groups"] = ["10.1234/PAPER-A"]
        second["evidence"][0]["source_group"] = "10.1234/PAPER-A"
    elif mode == "paper":
        second["paper_scope"] = ["paper-a"]
        second["evidence"][0]["paper_id"] = "paper-a"
    elif mode == "template":
        second["origin_draft_id"] = first["origin_draft_id"]
    else:
        second[mode] = first[mode]
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_rows(a, [first])
    write_rows(b, [second])
    assert load_cases(a) and load_cases(b)
    with pytest.raises(AnnotationError, match="split_leakage|annotation_dataset_duplicate"):
        audit_case_collection([a, b])


def test_source_separated_splits_are_structurally_valid_but_not_certified_independent(tmp_path):
    first, second = pair(tmp_path)
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_rows(a, [first])
    write_rows(b, [second])
    report = audit_case_collection([a, b])
    assert report["split_counts"] == {"development": 1, "holdout": 1}
    assert report["source_group_count"] == report["case_count"] == 2
    assert not report["independent_holdout_claimed"]
    assert [d["dataset_sha256"] for d in report["datasets"]] == [cases_hash(a), cases_hash(b)]


@pytest.mark.parametrize("drift", [False, True])
def test_cli_reference_check_does_not_score_references_or_hide_target_drift(tmp_path, monkeypatch, drift):
    from unittest.mock import Mock

    from evaluation import evidence_cases

    dataset, manifest = frozen(tmp_path)
    _, second = pair(tmp_path)
    reference, output = tmp_path / "reference.jsonl", tmp_path / "report.json"
    write_rows(reference, [second])
    monkeypatch.setattr(evidence_cases, "_ReadOnlyChunks", lambda _: Mock(get_chunks_by_ids=Mock(return_value=[canonical()])))
    monkeypatch.setattr(sys, "argv", ["evidence_cases", "--dataset", str(dataset), "--manifest", str(manifest),
                                     "--database", "unused", "--reference-dataset", str(reference), "--output", str(output)])
    if drift:
        original = evidence_cases.audit_case_collection

        def changed(paths):
            report = original(paths)
            report["datasets"][0]["dataset_sha256"] = "0" * 64
            return report

        monkeypatch.setattr(evidence_cases, "audit_case_collection", changed)
        with pytest.raises(SystemExit):
            evidence_cases.main()
        assert not output.exists()
    else:
        evidence_cases.main()
        report = json.loads(output.read_text("utf-8"))
        assert report["case_count"] == 1
        assert report["dataset_collection"]["case_count"] == 2
        assert "keyword_baseline" not in report


@pytest.mark.parametrize("count", [0, 17])
def test_collection_size_is_bounded_before_reading_files(count):
    with pytest.raises(AnnotationError, match="annotation_collection_size_invalid"):
        audit_case_collection([Path("missing.jsonl")] * count)


def test_repeated_file_path_is_rejected(tmp_path):
    a = tmp_path / "a.jsonl"
    write_rows(a, [case()])
    with pytest.raises(AnnotationError, match="annotation_collection_duplicate_path"):
        audit_case_collection([a, a])


def test_checked_in_expansion_has_four_new_tasks_and_original_pilot_is_unchanged():
    old = Path("data/evaluation/evidence_pilot_20261028.jsonl")
    new = Path("data/evaluation/evidence_expansion_20261029.jsonl")
    for path in (old, new):
        manifest = json.loads(path.with_suffix(".manifest.json").read_text("utf-8"))
        assert cases_hash(path) == manifest["dataset_sha256"]
    report = audit_case_collection([old, new])
    assert len(load_cases(old)) == 8 and len(load_cases(new)) == 4
    assert report["case_count"] == 12 and report["source_group_count"] == 4
    assert report["split_counts"] == {"development": 12}
    assert not report["independent_holdout_claimed"]
