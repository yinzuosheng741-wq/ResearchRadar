"""Read-only local corpus audit and keyword baseline; no network or model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from evaluation.dataset import load_questions
from evaluation.metrics import (
    deterministic_mean,
    evidence_group_recall_at_k,
    recall_at_k,
)
from retrieval.keyword_index import KeywordIndex


class ReadOnlyCorpus:
    def __init__(self, path: Path):
        if not path.is_file():
            raise ValueError("audit_database_missing")
        self.uri = path.resolve().as_uri() + "?mode=ro"

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.uri, uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        try:
            yield connection
        finally:
            connection.close()


def audit_corpus(database: Path, dataset: Path | None = None) -> dict:
    corpus = ReadOnlyCorpus(database)
    with corpus._connect() as connection:
        papers = connection.execute("SELECT paper_id, status FROM papers ORDER BY paper_id").fetchall()
        chunks = connection.execute(
            "SELECT chunk_id, paper_id, page_number, section, text FROM chunks ORDER BY chunk_id"
        ).fetchall()
    by_id = {row["chunk_id"]: row for row in chunks}
    paper_ids = {row["paper_id"] for row in papers}
    paper_chunks = Counter(row["paper_id"] for row in chunks)
    fulltext_papers = {row["paper_id"] for row in chunks if row["page_number"] > 0}
    fingerprints = [
        (row["chunk_id"], row["paper_id"], row["page_number"], row["section"],
         hashlib.sha256(row["text"].encode("utf-8")).hexdigest())
        for row in chunks
    ]
    duplicate_keys = Counter(
        (row["paper_id"], row["page_number"], " ".join(row["text"].split())) for row in chunks
    )
    payload = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "scope": "local_corpus_and_keyword_only",
        "network_calls": 0,
        "model_calls": 0,
        "corpus_fingerprint": hashlib.sha256(
            json.dumps(fingerprints, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "corpus": {
            "papers": len(papers), "chunks": len(chunks),
            "status_counts": dict(sorted(Counter(row["status"] for row in papers).items())),
            "papers_without_chunks": sum(row["paper_id"] not in paper_chunks for row in papers),
            "papers_with_fulltext_chunks": len(fulltext_papers),
            "papers_with_abstract_chunks_only": len(set(paper_chunks) - fulltext_papers),
            "abstract_chunks": sum(row["page_number"] == 0 for row in chunks),
            "empty_chunks": sum(not row["text"].strip() for row in chunks),
            "negative_page_chunks": sum(row["page_number"] < 0 for row in chunks),
            "orphan_chunks": sum(row["paper_id"] not in paper_ids for row in chunks),
            "chunks_without_section": sum(not row["section"] for row in chunks),
            "duplicate_text_chunks_same_page": sum(count - 1 for count in duplicate_keys.values()),
        },
        "limitations": [
            "No comparison with original PDF pages; OCR/table loss and truncation remain unverified.",
            "Existing annotated local dataset is a diagnostic set, not a new independent holdout.",
            "No embedding, reranker, answer quality or semantic entailment evaluation.",
        ],
    }
    if dataset is None:
        return payload
    questions = load_questions(dataset)
    index = KeywordIndex(corpus)
    cases, values = [], {5: [], 20: []}
    paper_values = {5: [], 20: []}
    for question in questions:
        missing = [key for key in question.relevant_chunk_ids if key not in by_id]
        mismatch = [key for group in question.evidence_groups for key in group.chunk_ids
                    if key in by_id and by_id[key]["paper_id"] != group.paper_id]
        eligible = not missing and not mismatch
        rankings, scores = {}, {}
        for k in (5, 20):
            found = index.search(question.question, k=k)
            ranked = [item.chunk_id for item in found]
            rankings[str(k)] = ranked
            if eligible:
                score = evidence_group_recall_at_k(ranked, [g.chunk_ids for g in question.evidence_groups], k)
                scores[str(k)] = score
                values[k].append(score)
                paper_values[k].append(recall_at_k([c.paper_id for c in found], question.relevant_paper_ids, k))
        if not eligible:
            diagnosis = "annotation_not_in_corpus_or_wrong_paper"
        elif scores["5"] == 1:
            diagnosis = "covered_at_5"
        elif scores["20"] > scores["5"]:
            diagnosis = "evidence_below_top_5"
        else:
            diagnosis = "required_evidence_not_in_top_20"
        cases.append({
            "question_id": question.question_id, "category": question.category,
            "eligible": eligible, "missing_chunk_ids": missing, "wrong_paper_chunk_ids": mismatch,
            "diagnosis": diagnosis, "ranked_chunk_ids": rankings, "evidence_group_recall": scores,
            "uncovered_groups_at_20": [
                group.paper_id for group in question.evidence_groups
                if not set(group.chunk_ids) & set(rankings["20"])
            ],
        })
    payload["dataset"] = {
        "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "questions": len(questions), "eligible_questions": len(values[5]),
        "excluded_questions": len(questions) - len(values[5]),
        "provenance": "existing_local_annotation_diagnostic_set",
    }
    payload["keyword_metrics"] = {
        f"evidence_group_recall_at_{k}": deterministic_mean(values[k]) if values[k] else None for k in values
    }
    payload["keyword_metrics"].update({
        f"paper_recall_at_{k}": deterministic_mean(paper_values[k]) if paper_values[k] else None for k in values
    })
    payload["diagnosis_counts"] = dict(sorted(Counter(case["diagnosis"] for case in cases).items()))
    payload["cases"] = cases
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/research.db"))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Never overwrite the live database, labels, or any existing report.
    if args.output.exists() or args.output.suffix.lower() != ".json":
        parser.error("audit_output_must_be_new_json")
    payload = audit_corpus(args.database, args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"corpus": payload["corpus"], "dataset": payload.get("dataset"),
                      "keyword_metrics": payload.get("keyword_metrics"),
                      "diagnosis_counts": payload.get("diagnosis_counts")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
