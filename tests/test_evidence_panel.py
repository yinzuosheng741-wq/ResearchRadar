from streamlit.testing.v1 import AppTest

from domain.models import AnswerCitation, ResearchConversationMessage
from tests.test_chat_workspace import APP
from web.evidence_panel import answer_export, unique_citations


def citation(paper="p1"):
    return AnswerCitation(paper_id=paper, chunk_id="chunk", title="Paper <script>alert(1)</script>",
                          page_number=0, quote="Abstract evidence about water quality.")


def test_export_deduplicates_evidence_and_marks_abstracts_without_active_html():
    message = ResearchConversationMessage(role="assistant", content="A bounded answer.",
                                           citations=[citation(), citation()])
    assert len(unique_citations(message.citations)) == 1
    exported = answer_export(message)
    assert exported.count("摘要证据") == 1
    assert "<script>" not in exported
    assert "语义验证" in exported


def test_reply_without_sources_does_not_export_a_claim_of_evidence_support():
    text = answer_export(ResearchConversationMessage(role="assistant", content="Hello"))
    assert "没有论文引用" in text


def make_app(tmp_path, monkeypatch, paper="p1"):
    monkeypatch.setenv("RESEARCH_CHAT_DB", str(tmp_path / "chat.sqlite3"))
    code = APP.replace('return {"papers": []}', 'return {"papers": [{"paper_id": "p1", "title": "Water paper"}]}')
    path = tmp_path / "app.py"
    path.write_text(code, encoding="utf-8")
    app = AppTest.from_file(str(path)).run(timeout=20)
    app.session_state["research_chat_transcript"] = [ResearchConversationMessage(
        role="assistant", content="Evidence answer.", tool_name="evidence_qa", citations=[citation(paper)],
    ).model_dump(mode="json")]
    return app.run(timeout=20)


def test_citation_action_applies_known_scope_before_widgets_are_created(tmp_path, monkeypatch):
    app = make_app(tmp_path, monkeypatch)
    assert not app.exception
    app.button(key="evidence_scope_0_0").click().run(timeout=20)
    assert not app.exception
    assert app.radio(key="research_scope_mode").value == "仅选定论文"
    assert app.multiselect(key="research_paper_scope").value == ["p1"]


def test_stale_citation_scope_stays_empty_instead_of_widening_to_all_papers(tmp_path, monkeypatch):
    app = make_app(tmp_path, monkeypatch, paper="deleted")
    app.button(key="evidence_scope_0_0").click().run(timeout=20)
    assert not app.exception
    assert app.radio(key="research_scope_mode").value == "仅选定论文"
    assert app.multiselect(key="research_paper_scope").value == []
