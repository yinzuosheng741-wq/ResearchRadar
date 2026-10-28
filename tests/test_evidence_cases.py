"""Evidence drift, dataset leakage and honest offline metric denominators."""

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from domain.models import EvidenceChunk
from evaluation.evidence_cases import (
    AnnotationError,
    _ReadOnlyChunks,
    audit_cases,
    cases_hash,
    load_cases,
    retrieval_baseline,
    text_hash,
)


def case():
    return {"case_id": "pilot-1", "origin_draft_id": "draft-1", "question": "What does the selected paper measure?",
            "original_question": "What did the original study measure?", "split": "development",
            "answerability": "answerable", "expected_answer": "The selected paper measured chlorophyll.",
            "answer_limitations": "This does not prove independent validation.", "paper_scope": ["paper-a"],
            "source_groups": ["10.1234/paper-a"],
            "evidence": [{"chunk_id": "chunk-a", "paper_id": "paper-a", "page_number": 2,
                          "chunk_sha256": text_hash("The paper measured chlorophyll."),
                          "quote": "measured chlorophyll", "source_group": "10.1234/paper-a",
                          "pdf_sha256": hashlib.sha256(b"synthetic PDF fingerprint fixture").hexdigest(),
                          "locator": "PDF p2 / Methods"}],
            "review_method": "codex_original_page_and_chunk_comparison",
            "provenance": "adapted_from_existing_drafts_after_corpus_inspection"}


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), "utf-8")


def frozen(tmp_path, rows=None):
    dataset, manifest = tmp_path / "cases.jsonl", tmp_path / "manifest.json"
    write_rows(dataset, rows or [case()])
    manifest.write_text(json.dumps({"schema_version": 1, "evaluation_role": "development_pilot",
                                   "reviewer": "synthetic-fixture", "reviewed_at": "2026-10-28",
                                   "dataset_sha256": cases_hash(dataset), "case_count": len(rows or [case()]),
                                   "split_counts": {"development": len(rows or [case()])},
                                   "independent_holdout_claimed": False}), "utf-8")
    return dataset, manifest


def canonical():
    return EvidenceChunk(chunk_id="chunk-a", paper_id="paper-a", page_number=2,
                         title="Synthetic paper", text="The paper measured chlorophyll.")


def test_matching_evidence_and_pdf_fingerprints_are_integrity_only(tmp_path):
    dataset, manifest = frozen(tmp_path)
    (tmp_path / "paper-a.pdf").write_bytes(b"synthetic PDF fingerprint fixture")
    report = audit_cases(dataset, manifest, Mock(get_chunks_by_ids=Mock(return_value=[canonical()])), papers_dir=tmp_path)
    assert report["status"] == "consistent"
    assert report["pdfs_checked"] == report["unique_chunks_checked"] == 1
    assert not report["independent_holdout_claimed"]
    assert report["scope"] == "annotation_integrity_not_model_quality"


@pytest.mark.parametrize("change,code", [
    ({"paper_id": "other"}, "annotation_evidence_identity_mismatch"),
    ({"page_number": 3}, "annotation_evidence_identity_mismatch"),
    ({"text": "Changed canonical evidence."}, "annotation_evidence_text_drift"),
])
def test_evidence_identity_or_text_drift_is_rejected(tmp_path, change, code):
    dataset, manifest = frozen(tmp_path)
    store = Mock(get_chunks_by_ids=Mock(return_value=[canonical().model_copy(update=change)]))
    with pytest.raises(AnnotationError, match=code):
        audit_cases(dataset, manifest, store)


@pytest.mark.parametrize("mode,code", [("missing", "annotation_evidence_missing"),
                                      ("duplicate", "annotation_canonical_duplicate"),
                                      ("outage", "annotation_corpus_unavailable")])
def test_corpus_failure_is_not_certified_as_an_empty_label(tmp_path, mode, code):
    dataset, manifest = frozen(tmp_path)
    store = Mock(get_chunks_by_ids=Mock(return_value=[] if mode == "missing" else [canonical(), canonical()]))
    if mode == "outage":
        store.get_chunks_by_ids.side_effect = RuntimeError("private provider payload")
    with pytest.raises(AnnotationError, match=code):
        audit_cases(dataset, manifest, store)


def test_a_valid_chunk_fingerprint_does_not_make_an_invented_quote_valid(tmp_path):
    row = case()
    row["evidence"][0]["quote"] = "This invented result is supported."
    dataset, manifest = frozen(tmp_path, [row])
    with pytest.raises(AnnotationError, match="annotation_quote_mismatch"):
        audit_cases(dataset, manifest, Mock(get_chunks_by_ids=Mock(return_value=[canonical()])))


def test_changed_pdf_and_changed_label_both_invalidate_snapshot(tmp_path):
    dataset, manifest = frozen(tmp_path)
    (tmp_path / "paper-a.pdf").write_bytes(b"another PDF")
    store = Mock(get_chunks_by_ids=Mock(return_value=[canonical()]))
    with pytest.raises(AnnotationError, match="annotation_pdf_drift"):
        audit_cases(dataset, manifest, store, papers_dir=tmp_path)
    row = case()
    row["expected_answer"] = "A changed expected answer after freezing."
    write_rows(dataset, [row])
    with pytest.raises(AnnotationError, match="annotation_manifest_drift"):
        audit_cases(dataset, manifest, store)


@pytest.mark.parametrize("mode", ["doi", "paper", "template"])
def test_no_source_paper_or_draft_template_can_cross_splits(tmp_path, mode):
    first, second = case(), case()
    second.update(case_id="pilot-2", origin_draft_id="draft-2", question="A different question about the paper?", split="holdout")
    if mode == "doi":
        second["paper_scope"] = ["paper-b"]
        second["source_groups"] = ["10.1234/PAPER-A"]
        second["evidence"][0].update(chunk_id="chunk-b", paper_id="paper-b", source_group="10.1234/PAPER-A")
    elif mode == "paper":
        second["source_groups"] = ["10.1234/alias-doi"]
        second["evidence"][0]["source_group"] = "10.1234/alias-doi"
    else:
        second.update(origin_draft_id="draft-1", paper_scope=["paper-b"], source_groups=["10.1234/paper-b"])
        second["evidence"][0].update(chunk_id="chunk-b", paper_id="paper-b", source_group="10.1234/paper-b")
    dataset = tmp_path / "cases.jsonl"
    write_rows(dataset, [first, second])
    with pytest.raises(AnnotationError, match="split_leakage"):
        load_cases(dataset)


@pytest.mark.parametrize("mode", ["empty_gold", "unsupported_absence", "out_of_scope", "blank_answer", "duplicate_anchor"])
def test_incomplete_or_inconsistent_gold_labels_are_rejected(tmp_path, mode):
    row = case()
    if mode == "empty_gold":
        row["evidence"] = []
    elif mode == "unsupported_absence":
        row.update(answerability="insufficient_evidence", evidence=[])
    elif mode == "out_of_scope":
        row["paper_scope"] = ["another-paper"]
    elif mode == "blank_answer":
        row["expected_answer"] = " " * 20
    else:
        row["evidence"].append(dict(row["evidence"][0]))
    dataset = tmp_path / "cases.jsonl"
    write_rows(dataset, [row])
    with pytest.raises(AnnotationError, match="annotation_dataset_invalid"):
        load_cases(dataset)


def test_semantic_fingerprint_handles_crlf_json_key_order_and_extra_spaces(tmp_path):
    dataset, _ = frozen(tmp_path)
    expected = cases_hash(dataset)
    dataset.write_bytes((json.dumps(case(), sort_keys=True) + "\r\n").encode("utf-8"))
    assert cases_hash(dataset) == expected


def test_missing_readonly_database_is_not_created(tmp_path):
    target = tmp_path / "missing.sqlite3"
    with pytest.raises(AnnotationError, match="annotation_database_unavailable"):
        _ReadOnlyChunks(target)
    assert not target.exists()


def test_failed_retrieval_stays_in_denominator_and_clarifications_are_not_scored(tmp_path):
    first, second, clarification = case(), case(), case()
    second.update(case_id="pilot-2", question="Another evidence question about the paper?")
    clarification.update(case_id="pilot-3", question="Which unspecified paper did you mean?", answerability="needs_clarification",
                         evidence=[], paper_scope=None, review_method="codex_ambiguity_review")
    dataset = tmp_path / "cases.jsonl"
    write_rows(dataset, [first, second, clarification])
    retriever = Mock(search=Mock(side_effect=[[canonical()], RuntimeError("private provider payload")]))
    report = retrieval_baseline(load_cases(dataset), retriever)
    assert report["evaluated_cases"] == 2 and report["excluded_non_retrieval_cases"] == 1
    assert report["retrieval_failed_cases"] == 1 and report["mean_required_anchor_recall"] == 0.5
    assert all(call.kwargs["paper_ids"] == ["paper-a"] for call in retriever.search.call_args_list)
    assert "private" not in str(report)


def test_checked_in_pilot_is_eight_development_cases_and_has_matching_manifest():
    path = Path("data/evaluation/evidence_pilot_20261028.jsonl")
    cases = load_cases(path)
    manifest = json.loads(path.with_suffix(".manifest.json").read_text("utf-8"))
    assert len(cases) == manifest["case_count"] == 8
    assert {case.split for case in cases} == {"development"}
    assert sum(case.answerability == "answerable" for case in cases) == 6
    assert manifest["dataset_sha256"] == cases_hash(path)
