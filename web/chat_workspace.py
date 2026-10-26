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
                "research_paper_scope", "research_scope_mode"):
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
        st.divider()
        if st.button("＋ 新对话", key="research_new_chat", use_container_width=True):
            reset_chat(st)
            st.rerun()
        query = st.text_input("搜索历史对话", placeholder="按标题查找", key="research_history_query")
        entries = store.list(query)
        st.caption("历史对话 · 保存在本机")
        for item in entries[:40]:
            if st.button(item["title"], key=f"research_history_{item['id']}",
                         type="primary" if item["id"] == current else "secondary",
                         use_container_width=True):
                reset_chat(st, item["id"])
                st.rerun()
        if not entries:
            st.caption("暂无匹配的对话")


def render_memory(st, services, conversation):
    with st.expander("记忆与检索范围", expanded=False):
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
    st.markdown("""<style>
    [data-testid="stSidebar"] {background:#f7f7f8;border-right:1px solid #ececec;}
    [data-testid="stToolbar"] {display:none;}
    [data-testid="stDecoration"] {display:none;}
    .block-container {max-width:1000px;padding-top:2rem;padding-bottom:6rem;}
    [data-testid="stChatMessage"] {border:0!important;border-radius:18px!important;background:#fafafa;line-height:1.8;}
    [data-testid="stChatInput"] {border-radius:24px!important;border:1px solid #dedede;box-shadow:0 3px 18px #00000008;}
    .agent-welcome {border:0;background:transparent;text-align:center;padding:3rem 1rem;}
    .agent-welcome-title {font-size:1.9rem;color:#252525;}
    .agent-suggestion {border-radius:14px;background:#f3f4f6;color:#48515c;}
    @media(max-width:700px){.block-container{padding-left:1rem;padding-right:1rem;}}
    </style>""", unsafe_allow_html=True)
