from streamlit.testing.v1 import AppTest


APP = '''
from agent.research_agent import ResearchAgentService
from domain.models import ResearchProjectMemory
from web.app import DefaultUiServices, run_app
class Offline(DefaultUiServices):
    def project_memory(self): return ResearchProjectMemory()
    def knowledge_snapshot(self): return {"papers": []}
    def research_chat(self, message, conversation=None, *, execution_context=None):
        import streamlit as st
        st.session_state["received_paper_scope"] = None if execution_context is None else execution_context.paper_ids
        return ResearchAgentService(model=None, qa_service=None, plan_service=None).chat(
            message, conversation, execution_context=execution_context)
run_app(services=Offline())
'''


def test_workspace_saves_switches_and_forgets_context(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_CHAT_DB", str(tmp_path / "chats.sqlite3"))
    entry = tmp_path / "offline_app.py"
    entry.write_text(APP, encoding="utf-8")
    app = AppTest.from_file(str(entry)).run(timeout=20)
    assert not app.exception
    first_id = app.session_state["research_chat_id"]
    app.chat_input(key="research_agent_input").set_value("你好").run(timeout=20)
    assert not app.exception
    assert len(app.session_state["research_chat_transcript"]) == 2
    app.button(key="research_new_chat").click().run(timeout=20)
    assert app.session_state["research_chat_id"] != first_id
    assert app.session_state["research_chat_transcript"] == []
    app.button(key=f"research_history_{first_id}").click().run(timeout=20)
    assert len(app.session_state["research_chat_transcript"]) == 2
    app.button(key="research_forget_context").click().run(timeout=20)
    assert app.session_state["research_agent_conversation"]["messages"] == []
    assert len(app.session_state["research_chat_transcript"]) == 2
    assert not app.exception


def test_explicit_empty_paper_selection_does_not_widen_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_CHAT_DB", str(tmp_path / "chats.sqlite3"))
    entry = tmp_path / "offline_app.py"
    entry.write_text(APP, encoding="utf-8")
    app = AppTest.from_file(str(entry)).run(timeout=20)
    app.radio(key="research_scope_mode").set_value("仅选定论文").run(timeout=20)
    assert app.multiselect(key="research_paper_scope").value == []
    app.chat_input(key="research_agent_input").set_value("你好").run(timeout=20)
    assert not app.exception
    assert app.multiselect(key="research_paper_scope").value == []
    assert app.session_state["received_paper_scope"] == ()
