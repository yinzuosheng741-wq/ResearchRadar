import json
import sqlite3

import pytest

from domain.models import EvidenceChunk, PaperCandidate
from evaluation.corpus_audit import ReadOnlyCorpus, audit_corpus
from storage.database import ResearchDatabase


def test_audit_excludes_stale_labels_and_distinguishes_abstract_and_fulltext(tmp_path):
    path = tmp_path / "corpus.db"
    database = ResearchDatabase(path)
    paper = database.upsert_candidate(PaperCandidate(source="local", source_id="one", title="Water evidence"))
    abstract, fulltext = f"{paper.paper_id}:a", f"{paper.paper_id}:b"
    database.replace_chunks(paper.paper_id, [
        EvidenceChunk(chunk_id=abstract, paper_id=paper.paper_id, title="Water evidence", page_number=0, text="water abstract"),
        EvidenceChunk(chunk_id=fulltext, paper_id=paper.paper_id, title="Water evidence", page_number=2, text="water evidence"),
    ])
    labels = tmp_path / "questions.jsonl"
    labels.write_text("\n".join(json.dumps({
        "question_id": key, "question": "water", "evidence_groups": [{
            "paper_id": paper.paper_id, "chunk_ids": [cid], "rationale": "Direct evidence."
        }]
    }) for key, cid in (("valid", fulltext), ("stale", f"{paper.paper_id}:missing"))), encoding="utf-8")
    before = path.read_bytes()
    result = audit_corpus(path, labels)
    assert path.read_bytes() == before
    assert result["corpus"]["papers_with_fulltext_chunks"] == 1
    assert result["corpus"]["papers_with_abstract_chunks_only"] == 0
    assert result["dataset"]["eligible_questions"] == 1
    assert result["dataset"]["excluded_questions"] == 1
    assert result["keyword_metrics"]["evidence_group_recall_at_5"] == 1
    assert result["diagnosis_counts"]["annotation_not_in_corpus_or_wrong_paper"] == 1
    assert "water abstract" not in json.dumps(result)


def test_audit_connection_rejects_writes_and_missing_database(tmp_path):
    path = tmp_path / "existing.db"
    ResearchDatabase(path)
    with ReadOnlyCorpus(path)._connect() as connection:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM chunks")
    with pytest.raises(ValueError, match="audit_database_missing"):
        ReadOnlyCorpus(tmp_path / "missing.db")
