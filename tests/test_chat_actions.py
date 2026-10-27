from domain.models import AnswerCitation, ResearchConversationMessage
from streamlit.testing.v1 import AppTest
from web.chat_actions import conversation_export

from tests.test_chat_workspace import APP


def app_at(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_CHAT_DB", str(tmp_path / "chat.sqlite3"))
    path = tmp_path / "app.py"
    path.write_text(APP, encoding="utf-8")
    return AppTest.from_file(str(path)).run(timeout=20)


def test_task_draft_does_not_call_research_until_user_edits_and_sends(tmp_path, monkeypatch):
    app = app_at(tmp_path, monkeypatch)
    app.button(key="research_prompt_comparison").click().run(timeout=20)
    assert not app.exception
    assert app.session_state["research_chat_transcript"] == []
    assert app.chat_input(key="research_agent_input").disabled
    app.text_area(key="research_task_text").set_value("你好，先说明你的用途")
    app.button(key="FormSubmitter:research_task_form-发送这个问题").click().run(timeout=20)
    assert not app.exception
    messages = app.session_state["research_chat_transcript"]
    assert len(messages) == 2 and messages[0]["content"] == "你好，先说明你的用途"
    assert not app.chat_input(key="research_agent_input").disabled


def test_empty_cancel_and_new_chat_never_execute_draft(tmp_path, monkeypatch):
    app = app_at(tmp_path, monkeypatch)
    app.button(key="research_prompt_plan").click().run(timeout=20)
    app.text_area(key="research_task_text").set_value("  ")
    app.button(key="FormSubmitter:research_task_form-发送这个问题").click().run(timeout=20)
    assert app.session_state["research_chat_transcript"] == []
    assert any("请先填写" in item.value for item in app.warning)
    app.button(key="FormSubmitter:research_task_form-取消").click().run(timeout=20)
    assert not app.text_area
    app.button(key="research_prompt_evidence").click().run(timeout=20)
    app.button(key="research_new_chat").click().run(timeout=20)
    assert not app.exception and not app.text_area
    assert not app.chat_input(key="research_agent_input").disabled


def test_conversation_export_keeps_both_roles_and_sanitizes_citations():
    citation = AnswerCitation(paper_id="p", chunk_id="c", title="<img src=x>", page_number=0, quote="Abstract")
    text = conversation_export([
        ResearchConversationMessage(role="user", content="你好"),
        ResearchConversationMessage(role="assistant", content="Hello", citations=[citation, citation]),
    ])
    assert "你好" in text and "Hello" in text
    assert text.count("摘要证据") == 1
    assert "<img" not in text and "语义验证" in text


def test_explicit_empty_scope_is_visible_outside_settings(tmp_path, monkeypatch):
    app = app_at(tmp_path, monkeypatch)
    app.radio(key="research_scope_mode").set_value("仅选定论文").run(timeout=20)
    assert any("仅选定 0 篇论文" in item.value for item in app.caption)
    assert any("不会检索证据" in item.value for item in app.warning)


def test_missing_model_shows_actionable_notice_and_keeps_edited_draft(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_CHAT_DB", str(tmp_path / "chat.sqlite3"))
    code = APP.replace("run_app(services=Offline())", '''
class Missing(Offline):
    def research_chat(self, *args, **kwargs):
        raise ValueError("research_model_not_configured")
run_app(services=Missing())
''')
    path = tmp_path / "missing_model.py"
    path.write_text(code, encoding="utf-8")
    app = AppTest.from_file(str(path)).run(timeout=20)
    app.button(key="research_prompt_plan").click().run(timeout=20)
    app.text_area(key="research_task_text").set_value("请规划一个叶绿素研究路线")
    app.button(key="FormSubmitter:research_task_form-发送这个问题").click().run(timeout=20)
    assert not app.exception
    assert app.session_state["research_chat_transcript"] == []
    assert app.text_area(key="research_task_text").value == "请规划一个叶绿素研究路线"
    assert any("尚未配置对话模型" in item.value for item in app.info)
