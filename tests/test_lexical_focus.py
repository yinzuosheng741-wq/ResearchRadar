"""Bilingual facets, rank scales, candidate budgets and immutable paper scope."""

from contextlib import nullcontext
from unittest.mock import Mock

import pytest
from domain.models import EvidenceChunk, PaperCandidate
from retrieval.keyword_index import KeywordIndex
from retrieval.lexical_plan import lexical_focus
from storage.database import ResearchDatabase


def index(tmp_path):
    db = ResearchDatabase(tmp_path / "lexical.sqlite3")
    paper = db.upsert_candidate(PaperCandidate(source="test", source_id="instrument", title="SpectrumKit field dataset"))
    db.replace_chunks(paper.paper_id, [
        EvidenceChunk(chunk_id="spectra", paper_id=paper.paper_id, title=paper.title, page_number=2,
                      text="Reflectance spectra have sampling intervals of 2 nm across the wavelength range 450 to 700 nm."),
        EvidenceChunk(chunk_id="uncertainty", paper_id=paper.paper_id, title=paper.title, page_number=3,
                      text="Uncertainty estimates require methodological details, some of which were not recorded."),
        EvidenceChunk(chunk_id="generic", paper_id=paper.paper_id, title=paper.title, page_number=4,
                      text="Methods methods methods. Dataset data table instrument methods catalogue."),
    ])
    return KeywordIndex(db), paper


@pytest.mark.parametrize("query,expected", [("SpectrumKit 光谱采样间隔是多少", "spectra"),
                                          ("SpectrumKit 不确定性与方法细节", "uncertainty")])
def test_bilingual_aspects_beat_generic_title_and_methods_terms(tmp_path, query, expected):
    search, paper = index(tmp_path)
    rows = search.search(query, 2, paper_ids=[paper.paper_id])
    assert rows[0].chunk_id == expected
    assert len(rows) <= 2 and all(row.paper_id == paper.paper_id for row in rows)
    assert search.last_trace.ranking_mode == "focused_rrf"
    assert search.last_trace.sql_calls <= search.last_trace.planned_branches * 2 <= 10
    assert search.last_trace.candidate_window == 20


def test_focus_does_not_invent_satellite_names_or_numeric_values():
    plan = lexical_focus("某传感器的波段与水质变量")
    assert plan.entities == ()
    terms = {term for group in plan.concepts for term in group}
    assert "sensor" in terms
    assert not {"Sentinel-2", "Landsat", "MODIS", "350", "900"} & terms


def test_english_only_query_keeps_compatibility_path(tmp_path):
    search, _ = index(tmp_path)
    assert lexical_focus("spectral sampling intervals").concepts == ()
    assert search.search("spectral sampling intervals", 2)[0].chunk_id == "spectra"
    assert search.last_trace.ranking_mode == "legacy_bm25"


def test_empty_scope_has_no_sql_and_resets_previous_diagnostics(tmp_path):
    search, _ = index(tmp_path)
    search.search("SpectrumKit 光谱间隔", 2)
    assert search.last_trace.sql_calls > 0
    search.database = Mock()
    assert search.search("SpectrumKit 光谱间隔", 2, paper_ids=[]) == []
    search.database._connect.assert_not_called()
    assert search.last_trace.sql_calls == search.last_trace.selected_count == 0


def test_generated_expressions_cannot_turn_user_tokens_into_fts_operators(tmp_path):
    search, paper = index(tmp_path)
    assert search.search('SpectrumKit 光谱间隔 " OR 1=1 --', 4, paper_ids=["missing') OR 1=1 --"]) == []
    assert search.search("SpectrumKit 光谱间隔", 4, paper_ids=[paper.paper_id])[0].chunk_id == "spectra"


def test_focused_rrf_ignores_cross_query_bm25_magnitudes():
    def run(scale):
        def row(key, rank):
            return {"chunk_id": key, "paper_id": "p", "title": key, "page_number": 1,
                    "section": None, "text": key, "rank": rank}

        def execute(sql, parameters):
            expression = parameters[0]
            if "uncertainty" in expression:
                values = [row("focused", -0.001), row("other", -0.0001)]
            elif " AND " in expression:
                values = []
            else:
                values = [row("broad", -scale), row("focused", -scale / 2)]
            return Mock(fetchall=Mock(return_value=values))

        connection = Mock(execute=execute)
        search = KeywordIndex(Mock(_connect=Mock(return_value=nullcontext(connection))))
        return [(r.chunk_id, r.score) for r in search.search("SpectrumKit 不确定性", 3)]

    assert run(10) == run(10 ** 12)
    assert run(10)[0][0] == "focused"
