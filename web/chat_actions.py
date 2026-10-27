"""Explicit task drafts and transcript exports; neither invokes research tools."""

from web.evidence_panel import answer_export
from web.presenters import escape_untrusted


def conversation_export(messages):
    parts = ["# ResearchRadar 对话", "", "导出可阅读历史及其引用；不包含内部记忆或完整论文。", ""]
    for message in messages:
        if message.role == "assistant":
            answer = answer_export(message).removeprefix("# 研究回答\n\n").replace("\n## 引用证据\n", "\n### 引用证据\n")
            parts.extend(["## ResearchRadar", "", answer, ""])
        else:
            parts.extend(["## 你", "", escape_untrusted(message.content), ""])
    return "\n".join(parts)


def start_task_draft(state, text):
    state["research_task_draft"] = True
    state["research_task_text"] = text


def render_task_draft(st):
    if not st.session_state.get("research_task_draft"):
        return None
    with st.container(key="workspace_draft"):
        st.caption("任务草稿 · 可补充研究对象、论文和比较维度")
        with st.form("research_task_form"):
            text = st.text_area("编辑你的问题", key="research_task_text", height=140, max_chars=4000)
            submit, cancel = st.columns(2)
            sent = submit.form_submit_button("发送这个问题", type="primary", use_container_width=True)
            cancelled = cancel.form_submit_button("取消", use_container_width=True)
        if cancelled:
            st.session_state.pop("research_task_draft", None)
            st.rerun()
        if sent:
            if not text.strip():
                st.warning("请先填写问题。")
            else:
                return text.strip()
    return None
