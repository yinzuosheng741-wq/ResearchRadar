import json
import pickle
import subprocess
import sys
from types import SimpleNamespace

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from pypdf import PdfWriter
import pytest

import evaluation.fulltext_quality as quality
from evaluation.fulltext_quality import RestrictedDocstoreUnpickler, audit_index, audit_pdf
from rag.vector_store import VectorStoreService


def _persist_index(path, *, text="Canonical water evidence", paper_id="paper"):
    metadata = {"chunk_id": "chunk", "paper_id": paper_id, "page_number": 1, "title": "Water"}
    document = VectorStoreService._with_embedding_text(Document(page_content=text, metadata=metadata))
    with (path / "index.pkl").open("wb") as handle:
        pickle.dump((InMemoryDocstore({"chunk": document}), {0: "chunk"}), handle)
    (path / "index.faiss").write_bytes(b"index loading is isolated from payload unit tests")
    (path / "vector_store_manifest.json").write_text(json.dumps({"document_count": 1}), encoding="utf-8")


def test_index_audit_handles_enriched_embedding_text_and_detects_stale_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(quality, "_load_index", lambda _: SimpleNamespace(ntotal=1, d=2))
    _persist_index(tmp_path)
    chunks = [{"chunk_id": "chunk", "paper_id": "paper", "page_number": 1, "text": "Canonical water evidence"}]
    assert audit_index(chunks, tmp_path)["status"] == "pass"
    chunks[0]["text"] = "Updated canonical evidence"
    report = audit_index(chunks, tmp_path)
    assert report["status"] == "fail"
    assert report["payload_mismatches"] == ["chunk"]
    assert report["missing_chunk_ids"] == []


def test_index_audit_detects_wrong_scope_and_missing_or_stale_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(quality, "_load_index", lambda _: SimpleNamespace(ntotal=1, d=2))
    _persist_index(tmp_path, paper_id="other-paper")
    rows = [{"chunk_id": "chunk", "paper_id": "paper", "page_number": 1, "text": "Canonical water evidence"}]
    assert audit_index(rows, tmp_path)["payload_mismatches"] == ["chunk"]
    rows[0]["chunk_id"] = "new"
    report = audit_index(rows, tmp_path)
    assert report["missing_chunk_ids"] == ["new"]
    assert report["stale_chunk_ids"] == ["chunk"]


def test_docstore_audit_refuses_unexpected_pickle_globals(tmp_path):
    path = tmp_path / "untrusted.pkl"
    path.write_bytes(pickle.dumps(eval))
    with path.open("rb") as handle, pytest.raises(pickle.UnpicklingError, match="class_not_allowed"):
        RestrictedDocstoreUnpickler(handle).load()


def test_pdf_audit_flags_out_of_range_page_and_never_approves_semantics(tmp_path):
    path = tmp_path / "paper.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with path.open("wb") as handle:
        writer.write(handle)
    before = path.read_bytes()
    report = audit_pdf({"paper_id": "p", "title": "Water"}, [
        {"chunk_id": "bad", "page_number": 2, "text": "Wrong page", "section": None},
    ], path)
    assert path.read_bytes() == before
    assert report["invalid_page_numbers"] == [2]
    assert not report["fulltext_approved"]
    assert report["semantic_review"] == "pending"


def test_real_faiss_reader_supports_unicode_path_in_an_isolated_process(tmp_path):
    # Windows native DLL runtimes conflict when loaded alongside the full test
    # suite's ML libraries. Exercise the actual index reader in a clean process.
    script = """
import sys
from pathlib import Path
import faiss
import numpy as np
from evaluation.fulltext_quality import _load_index
path = Path(sys.argv[1]) / '\u5168\u6587\u7d22\u5f15.faiss'
index = faiss.IndexFlatL2(2)
index.add(np.array([[0, 1]], dtype='float32'))
path.write_bytes(faiss.serialize_index(index).tobytes())
loaded = _load_index(path)
assert loaded.ntotal == 1 and loaded.d == 2
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
