"""Desktop conversation navigation and local memory controls."""

import os
from pathlib import Path
from uuid import uuid4

from storage.conversations import ConversationStore

STATE_KEY = "research_agent_conversation"
TRANSCRIPT_KEY = "research_chat_transcript"


def reset_chat(st, conversation_id=None):
    st.session_state["research_chat_id"] = conversation_id or uuid4().hex
    st.session_state["research_navigation_target"] = "科研助手"
    for key in (STATE_KEY, TRANSCRIPT_KEY, "research_agent_pending_search",
                "research_paper_scope", "research_scope_mode", "research_scope_request", "research_delete_requested",
                "research_task_draft", "research_task_text"):
        st.session_state.pop(key, None)


def render_workspace(st):
    path = Path(os.environ.get("RESEARCH_CHAT_DB") or
                Path(__file__).resolve().parents[1] / "data" / "desktop" / "conversations.sqlite3")
    store = ConversationStore(path)
    st.session_state["research_chat_store"] = store
    if "research_chat_id" not in st.session_state:
        st.session_state["research_chat_id"] = uuid4().hex
    current = st.session_state["research_chat_id"]
    if STATE_KEY not in st.session_state:
        state, transcript = store.load(current)
        st.session_state[STATE_KEY] = state.model_dump(mode="json")
        st.session_state[TRANSCRIPT_KEY] = [m.model_dump(mode="json") for m in transcript]
    with st.sidebar:
        if st.button("＋ 新对话", key="research_new_chat", use_container_width=True):
            reset_chat(st)
            st.rerun()
        query = st.text_input("搜索历史对话", placeholder="搜索对话…", key="research_history_query", label_visibility="collapsed")
        entries = store.list(query)
        st.caption("最近对话 · 保存在本机")
        for item in entries[:40]:
            if st.button(item["title"], key=f"research_history_{item['id']}",
                         type="primary" if item["id"] == current else "secondary",
                         use_container_width=True):
                reset_chat(st, item["id"])
                st.rerun()
        if not entries:
            st.caption("暂无匹配的对话")


def render_memory(st, services, conversation):
    pending_scope = st.session_state.pop("research_scope_request", None)
    if pending_scope is not None:
        known = {paper["paper_id"] for paper in services.knowledge_snapshot().get("papers", [])}
        st.session_state["research_scope_mode"] = "仅选定论文"
        st.session_state["research_paper_scope"] = [key for key in pending_scope if key in known]
    with st.popover("研究范围与记忆", icon=":material/tune:", use_container_width=False):
        st.caption("对话记忆只用于理解问题；论文范围由你选择。新对话不会自动继承本次论文选择。")
        mode = st.radio("本次检索范围", ["全部可用论文", "仅选定论文"], horizontal=True,
                        key="research_scope_mode")
        if mode == "仅选定论文":
            papers = services.knowledge_snapshot().get("papers", [])
            titles = {p["paper_id"]: p.get("title", p["paper_id"]) for p in papers}
            st.multiselect("允许检索的论文", list(titles), format_func=lambda value: titles[value],
                           key="research_paper_scope")
            st.caption("未选论文时不检索任何证据，不会自动扩大范围。")
        if st.button("忘记本次对话上下文", key="research_forget_context"):
            store = st.session_state["research_chat_store"]
            store.forget_context(st.session_state["research_chat_id"])
            st.session_state.pop(STATE_KEY, None)
            st.session_state.pop("research_agent_pending_search", None)
            st.rerun()
        st.caption("遗忘上下文后，历史仍可阅读，但不会随下一次请求发给模型；项目记忆单独管理。")
        memory = services.project_memory()
        st.markdown("**已确认的项目记忆**")
        values = memory.model_dump(exclude={"facts", "updated_at", "last_active_skill", "project_id"})
        st.json({key: value for key, value in values.items() if value})
        if any(conversation.scope.model_dump().values()):
            st.caption("将上方“当前研究范围”保存为项目记忆后，新对话也可使用。")
            if st.button("确认保存当前研究范围", key="research_confirm_memory"):
                services.confirm_scope(conversation.scope)
                st.rerun()
        if st.button("清除项目记忆", key="research_forget_project"):
            services.clear_project_memory()
            st.rerun()


def inject_chat_style(st):
    css = Path(__file__).with_name("desktop.css").read_text(encoding="utf-8")
    st.markdown("<style>" + css + "</style>", unsafe_allow_html=True)
