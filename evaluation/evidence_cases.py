"""Frozen, source-grouped evidence annotations; integrity is not answer quality."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

from domain.models import EvidenceChunk
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from evaluation.corpus_audit import ReadOnlyCorpus
from evaluation.metrics import recall_at_k


class AnnotationError(ValueError):
    """Stable errors that never echo rejected dataset/provider content."""


class FrozenManifest(BaseModel):
    model_config = ConfigDict(extra="allow")  # Keep source attribution and review notes.
    schema_version: int = Field(strict=True, ge=1, le=1)
    evaluation_role: Literal["development_pilot", "reviewed_dataset"]
    dataset_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    case_count: int = Field(strict=True, ge=1)
    split_counts: dict[Literal["development", "holdout"], Annotated[int, Field(strict=True, ge=0)]]
    reviewer: str = Field(min_length=3)
    reviewed_at: date
    independent_holdout_claimed: Literal[False]


def text_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


class EvidenceAnchor(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    chunk_id: str = Field(min_length=1, max_length=180)
    paper_id: str = Field(min_length=1, max_length=120)
    page_number: int = Field(ge=1, strict=True)
    chunk_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    quote: str = Field(min_length=8, max_length=600)
    source_group: str = Field(pattern=r"^10\.\d{4,9}/[^\s]+$")
    pdf_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    locator: str = Field(min_length=3, max_length=240)


class EvidenceCase(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    case_id: str = Field(min_length=1, max_length=100)
    origin_draft_id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=8, max_length=4000)
    original_question: str = Field(min_length=8, max_length=4000)
    split: Literal["development", "holdout"]
    answerability: Literal["answerable", "partial", "needs_clarification", "insufficient_evidence"]
    expected_answer: str = Field(min_length=8, max_length=1600)
    answer_limitations: str = Field(min_length=8, max_length=1000)
    paper_scope: list[str] | None = Field(default=None, max_length=20)
    source_groups: list[str] = Field(min_length=1, max_length=20)
    evidence: list[EvidenceAnchor] = Field(default_factory=list, max_length=20)
    review_method: Literal["codex_original_page_and_chunk_comparison", "codex_ambiguity_review"]
    provenance: Literal["adapted_from_existing_drafts_after_corpus_inspection"]

    @model_validator(mode="after")
    def require_evidence_and_scope(self):
        if any(not value.strip() for value in self.source_groups) or len(set(self.source_groups)) != len(self.source_groups):
            raise ValueError("annotation_source_groups_invalid")
        if self.paper_scope is not None and (not self.paper_scope or len(set(self.paper_scope)) != len(self.paper_scope)):
            raise ValueError("annotation_paper_scope_invalid")
        if self.answerability in {"answerable", "partial"} and not self.evidence:
            raise ValueError("annotation_answer_requires_evidence")
        if self.answerability == "needs_clarification" and self.evidence:
            raise ValueError("annotation_clarification_has_no_gold_retrieval")
        if self.answerability == "insufficient_evidence":
            # Absence cannot be inferred from a few checked pages. Until an
            # explicit absence-search contract exists, refuse this gold label.
            raise ValueError("annotation_absence_review_not_supported")
        if self.evidence and self.paper_scope is None:
            raise ValueError("annotation_evidence_requires_explicit_scope")
        if len({anchor.chunk_id for anchor in self.evidence}) != len(self.evidence):
            raise ValueError("annotation_duplicate_anchor")
        for anchor in self.evidence:
            if anchor.paper_id not in self.paper_scope or anchor.source_group not in self.source_groups:
                raise ValueError("annotation_anchor_outside_scope")
        return self


def load_cases(path: Path) -> list[EvidenceCase]:
    try:
        if path.stat().st_size > 8_000_000:
            raise AnnotationError("annotation_dataset_too_large")
        lines = path.read_text("utf-8").splitlines()
        if len(lines) > 1000 or any(len(line.encode("utf-8")) > 65_536 for line in lines):
            raise AnnotationError("annotation_dataset_too_large")
        cases = [EvidenceCase.model_validate_json(line) for line in lines if line.strip()]
    except (OSError, UnicodeError, ValidationError, ValueError) as exc:
        if isinstance(exc, AnnotationError):
            raise
        raise AnnotationError("annotation_dataset_invalid") from None
    if not cases:
        raise AnnotationError("annotation_dataset_empty")
    if len({case.case_id for case in cases}) != len(cases) or len({" ".join(case.question.split()).casefold() for case in cases}) != len(cases):
        raise AnnotationError("annotation_dataset_duplicate")
    groups: dict[str, str] = {}
    papers: dict[str, tuple[str, str]] = {}
    scopes: dict[str, str] = {}
    origins: dict[str, str] = {}
    for case in cases:
        if case.origin_draft_id in origins and origins[case.origin_draft_id] != case.split:
            raise AnnotationError("annotation_template_split_leakage")
        origins[case.origin_draft_id] = case.split
        for paper in case.paper_scope or []:
            if paper in scopes and scopes[paper] != case.split:
                raise AnnotationError("annotation_paper_scope_split_leakage")
            scopes[paper] = case.split
        for group in case.source_groups:
            # DOI case differences must not allow the same paper into holdout.
            key = group.strip().casefold()
            if key in groups and groups[key] != case.split:
                raise AnnotationError("annotation_source_split_leakage")
            groups[key] = case.split
        for anchor in case.evidence:
            identity = (anchor.source_group.casefold(), case.split)
            if anchor.paper_id in papers and papers[anchor.paper_id] != identity:
                raise AnnotationError("annotation_paper_identity_or_split_mismatch")
            papers[anchor.paper_id] = identity
    return cases


def cases_hash(path: Path) -> str:
    """Semantic JSON fingerprint; stable across indentation, key order and CRLF."""
    return _cases_hash(load_cases(path))


def _cases_hash(cases: list[EvidenceCase]) -> str:
    payload = [case.model_dump(mode="json") for case in cases]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def audit_cases(dataset: Path, manifest: Path, chunk_store, *, papers_dir: Path | None = None) -> dict:
    cases = load_cases(dataset)
    fingerprint = _cases_hash(cases)
    try:
        frozen = FrozenManifest.model_validate_json(manifest.read_text("utf-8"))
        if frozen.dataset_sha256 != fingerprint:
            raise AnnotationError("annotation_manifest_drift")
        if frozen.case_count != len(cases) or frozen.split_counts != dict(Counter(c.split for c in cases)):
            raise AnnotationError("annotation_manifest_counts_mismatch")
        if frozen.evaluation_role == "development_pilot" and any(c.split == "holdout" for c in cases):
            raise AnnotationError("annotation_pilot_cannot_be_holdout")
    except AnnotationError:
        raise
    except (OSError, ValueError, TypeError, AttributeError):
        raise AnnotationError("annotation_manifest_invalid") from None
    anchors = [anchor for case in cases for anchor in case.evidence]
    keys = list(dict.fromkeys(anchor.chunk_id for anchor in anchors))
    try:
        found = chunk_store.get_chunks_by_ids(keys)
        canonical = {chunk.chunk_id: chunk for chunk in found}
        if len(canonical) != len(found):
            raise AnnotationError("annotation_canonical_duplicate")
    except AnnotationError:
        raise
    except Exception:
        raise AnnotationError("annotation_corpus_unavailable") from None
    pdf_hashes: dict[str, str] = {}
    for anchor in anchors:
        chunk = canonical.get(anchor.chunk_id)
        if chunk is None:
            raise AnnotationError("annotation_evidence_missing")
        if (chunk.paper_id, chunk.page_number) != (anchor.paper_id, anchor.page_number):
            raise AnnotationError("annotation_evidence_identity_mismatch")
        if text_hash(chunk.text) != anchor.chunk_sha256:
            raise AnnotationError("annotation_evidence_text_drift")
        if " ".join(anchor.quote.split()) not in " ".join(chunk.text.split()):
            raise AnnotationError("annotation_quote_mismatch")
        if papers_dir is not None:
            if anchor.paper_id not in pdf_hashes:
                # Treat file identity as data, never as a path supplied by a model.
                target = (papers_dir / (anchor.paper_id + ".pdf")).resolve()
                if target.parent != papers_dir.resolve():
                    raise AnnotationError("annotation_pdf_path_invalid")
                try:
                    pdf_hashes[anchor.paper_id] = hashlib.sha256(target.read_bytes()).hexdigest()
                except OSError:
                    raise AnnotationError("annotation_pdf_unavailable") from None
            if pdf_hashes[anchor.paper_id] != anchor.pdf_sha256:
                raise AnnotationError("annotation_pdf_drift")
    return {"schema_version": 1, "status": "consistent", "scope": "annotation_integrity_not_model_quality",
            "dataset_sha256": fingerprint, "case_count": len(cases),
            "split_counts": dict(Counter(c.split for c in cases)),
            "answerability_counts": dict(Counter(c.answerability for c in cases)),
            "unique_chunks_checked": len(keys), "pdfs_checked": len(pdf_hashes),
            "independent_holdout_claimed": False, "model_calls": 0,
            "limitations": ["Quote matching and fingerprints do not prove semantic answer support.",
                            "Review method and expected answer are annotations, not automatic expert certification."]}


class _ReadOnlyChunks:
    def __init__(self, database: Path):
        try:
            self.corpus = ReadOnlyCorpus(database)
        except ValueError:
            raise AnnotationError("annotation_database_unavailable") from None

    def get_chunks_by_ids(self, keys):
        if not keys:
            return []
        with self.corpus._connect() as connection:
            rows = connection.execute("SELECT * FROM chunks WHERE chunk_id IN (" + ",".join("?" for _ in keys) + ")", keys)
            return [EvidenceChunk.model_validate(dict(row)) for row in rows]


def retrieval_baseline(cases: list[EvidenceCase], retriever, *, k: int = 5) -> dict:
    """Score required anchors only; failed searches stay in the denominator."""
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 100:
        raise AnnotationError("annotation_retrieval_k_invalid")
    results = []
    for case in cases:
        if case.answerability not in {"answerable", "partial"}:
            continue
        failed = False
        try:
            candidates = retriever.search(case.question, k=k, paper_ids=list(case.paper_scope))
            ranked = list(dict.fromkeys(c.chunk_id for c in candidates if c.paper_id in case.paper_scope))[:k]
        except Exception:
            ranked, failed = [], True
        score = recall_at_k(ranked, [anchor.chunk_id for anchor in case.evidence], k)
        results.append({"case_id": case.case_id, "required_anchor_recall": score,
                        "ranked_chunk_ids": ranked, "retrieval_failed": failed})
    return {"scope": "development_pilot_required_anchors_not_exhaustive_relevance",
            "k": k, "evaluated_cases": len(results), "excluded_non_retrieval_cases": len(cases) - len(results),
            "retrieval_failed_cases": sum(row["retrieval_failed"] for row in results),
            "mean_required_anchor_recall": sum(row["required_anchor_recall"] for row in results) / len(results) if results else None,
            "cases": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--papers-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keyword-baseline", action="store_true", help="also run local scoped FTS5; no model or embeddings")
    args = parser.parse_args()
    try:
        report = audit_cases(args.dataset, args.manifest, _ReadOnlyChunks(args.database), papers_dir=args.papers_dir)
        if args.keyword_baseline:
            from retrieval.keyword_index import KeywordIndex

            verified_cases = load_cases(args.dataset)
            if _cases_hash(verified_cases) != report["dataset_sha256"]:
                raise AnnotationError("annotation_manifest_drift")
            report["keyword_baseline"] = retrieval_baseline(verified_cases, KeywordIndex(ReadOnlyCorpus(args.database)))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except (AnnotationError, OSError) as exc:
        parser.exit(1, f"{str(exc) if isinstance(exc, AnnotationError) else 'annotation_output_unavailable'}\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
