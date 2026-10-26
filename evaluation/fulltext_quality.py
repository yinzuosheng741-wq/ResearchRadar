"""Read-only PDF/chunk and persisted FAISS identity audit, without embeddings."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import pickle
import re

from pypdf import PdfReader

from evaluation.corpus_audit import ReadOnlyCorpus


class RestrictedDocstoreUnpickler(pickle.Unpickler):
    """Accept only the two data classes used by the local LangChain docstore."""

    def find_class(self, module: str, name: str):
        if (module, name) == ("langchain_core.documents.base", "Document"):
            from langchain_core.documents import Document

            return Document
        if (module, name) == ("langchain_community.docstore.in_memory", "InMemoryDocstore"):
            from langchain_community.docstore.in_memory import InMemoryDocstore

            return InMemoryDocstore
        raise pickle.UnpicklingError("audit_docstore_class_not_allowed")


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _normalized(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()


def _load_index(path: Path):
    import faiss
    import numpy as np

    return faiss.deserialize_index(np.frombuffer(path.read_bytes(), dtype="uint8"))


def audit_index(chunks: list[dict], store_path: Path) -> dict:
    """Compare canonical chunk identities, payloads, pages and FAISS counts."""
    with (store_path / "index.pkl").open("rb") as handle:
        docstore, mapping = RestrictedDocstoreUnpickler(handle).load()
    index = _load_index(store_path / "index.faiss")
    manifest = json.loads((store_path / "vector_store_manifest.json").read_text("utf-8"))
    canonical = {row["chunk_id"]: row for row in chunks}
    indexed: dict[str, object] = {}
    duplicates, missing_documents, id_mismatches, payload_mismatches = [], [], [], []
    for _, doc_id in sorted(mapping.items()):
        document = docstore.search(doc_id)
        if not hasattr(document, "metadata"):
            missing_documents.append(str(doc_id))
            continue
        metadata = document.metadata
        chunk_id = str(metadata.get("chunk_id", ""))
        if chunk_id in indexed:
            duplicates.append(chunk_id)
        indexed[chunk_id] = document
        if str(doc_id) != chunk_id:
            id_mismatches.append(str(doc_id))
        row = canonical.get(chunk_id)
        if row is not None and (
            str(metadata.get("paper_id")) != row["paper_id"]
            or metadata.get("page_number") != row["page_number"]
            or metadata.get("canonical_text", document.page_content) != row["text"]
        ):
            payload_mismatches.append(chunk_id)
    missing = sorted(canonical.keys() - indexed.keys())
    stale = sorted(indexed.keys() - canonical.keys())
    orphan_docstore_ids = sorted(set(docstore._dict) - set(mapping.values()))
    count_match = index.ntotal == len(mapping) == len(indexed) == manifest.get("document_count")
    position_match = set(mapping) == set(range(index.ntotal))
    passed = count_match and position_match and not any((
        missing, stale, duplicates, missing_documents, id_mismatches,
        payload_mismatches, orphan_docstore_ids,
    ))
    return {
        "status": "pass" if passed else "fail",
        "scope": "identity_payload_and_count_only_not_vector_quality",
        "canonical_count": len(canonical), "faiss_count": index.ntotal,
        "faiss_dimension": index.d, "mapping_count": len(mapping),
        "manifest": manifest, "positions_contiguous": position_match,
        "missing_chunk_ids": missing, "stale_chunk_ids": stale,
        "duplicate_chunk_ids": duplicates, "missing_documents": missing_documents,
        "docstore_id_mismatches": id_mismatches,
        "payload_mismatches": payload_mismatches,
        "orphan_docstore_ids": orphan_docstore_ids,
        "file_sha256": {name: _digest(store_path / name) for name in (
            "index.faiss", "index.pkl", "vector_store_manifest.json",
        )},
    }


def audit_pdf(paper: dict, chunks: list[dict], pdf_path: Path) -> dict:
    """Mechanical page alignment checks; semantic/table correctness needs review."""
    result = {
        "paper_id": paper["paper_id"], "title": paper["title"],
        "doi": paper.get("doi"), "license": paper.get("license"),
        "pdf_file": pdf_path.name, "chunk_count": len(chunks),
        "visual_review": "pending", "semantic_review": "pending",
    }
    if not pdf_path.is_file():
        return {**result, "status": "pdf_missing"}
    try:
        reader = PdfReader(pdf_path)
        texts = [(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:
        return {**result, "status": "pdf_read_failed", "error_type": type(exc).__name__}
    by_page: dict[int, list[dict]] = {}
    for chunk in chunks:
        by_page.setdefault(chunk["page_number"], []).append(chunk)
    pages, mismatches = [], []
    for number, text in enumerate(texts, 1):
        page_chunks = by_page.get(number, [])
        normalized = _normalized(text)
        for chunk in page_chunks:
            if _normalized(chunk["text"]) not in normalized:
                mismatches.append(chunk["chunk_id"])
        pages.append({
            "page_number": number, "extracted_characters": len(text.strip()),
            "chunk_count": len(page_chunks),
            "table_marker": bool(re.search(r"\btable\s+\d", text, re.I)),
            "method_marker": bool(re.search(r"\b(methods?|materials|methodology)\b", text, re.I)),
            "result_marker": bool(re.search(r"\b(results?|discussion)\b", text, re.I)),
            "replacement_characters": text.count("\ufffd"),
        })
    invalid = sorted(page for page in by_page if not 1 <= page <= len(texts))
    missing_pages = [p["page_number"] for p in pages
                     if p["extracted_characters"] >= 100 and not p["chunk_count"]]
    textless_pages = [p["page_number"] for p in pages if p["extracted_characters"] < 20]
    candidates = sorted(pages, key=lambda p: (
        -(2 * p["table_marker"] + p["method_marker"] + p["result_marker"]), p["page_number"],
    ))
    return {
        **result, "status": "needs_review" if invalid or missing_pages or mismatches or textless_pages else "alignment_pass",
        "pdf_sha256": _digest(pdf_path), "pdf_pages": len(texts), "pages": pages,
        "invalid_page_numbers": invalid, "text_pages_without_chunks": missing_pages,
        "textless_or_sparse_pages": textless_pages,
        "nonmatching_chunk_ids": mismatches,
        "chunks_without_section": sum(not c["section"] for c in chunks),
        "suggested_visual_page": candidates[0]["page_number"] if candidates else None,
        "fulltext_approved": False,
    }


def run_audit(database: Path, papers_dir: Path, store: Path, selection: Path) -> dict:
    with ReadOnlyCorpus(database)._connect() as connection:
        papers = [dict(row) for row in connection.execute("SELECT * FROM papers ORDER BY paper_id")]
        chunks = [dict(row) for row in connection.execute("SELECT * FROM chunks ORDER BY chunk_id")]
    selected = json.loads(selection.read_text("utf-8"))
    selected_ids = selected["paper_ids"]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("duplicate_selection_id")
    paper_map = {p["paper_id"]: p for p in papers}
    if set(selected_ids) - paper_map.keys():
        raise ValueError("unknown_selection_id")
    reviews = [audit_pdf(
        paper_map[paper_id],
        [c for c in chunks if c["paper_id"] == paper_id and c["page_number"] > 0],
        papers_dir / f"{paper_id}.pdf",
    ) for paper_id in selected_ids]
    return {
        "schema_version": 1, "timestamp_utc": datetime.now(UTC).isoformat(),
        "network_calls": 0, "model_calls": 0, "selection": selected,
        "index_audit": audit_index(chunks, store), "papers": reviews,
        "summary": dict(Counter(p["status"] for p in reviews)),
        "limitations": [
            "Text alignment is against pypdf extraction, not ground-truth PDF layout.",
            "Matching payloads do not prove embeddings were computed from those payloads.",
            "Titles and topic tags do not prove task coverage; visual and semantic reviews are separate.",
            "No source PDF, canonical database or vector index is modified.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/research.db"))
    parser.add_argument("--papers", type=Path, default=Path("data/papers"))
    parser.add_argument("--store", type=Path, default=Path("data/vector_store"))
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a new report path")
    report = run_audit(args.database, args.papers, args.store, args.selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"index": report["index_audit"]["status"], "papers": report["summary"]}))
    return int(report["index_audit"]["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
