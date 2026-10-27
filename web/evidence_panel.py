"""Evidence navigation and text export independent of research/model services."""

from domain.models import ResearchConversationMessage
from web.presenters import escape_untrusted, render_citation


def unique_citations(citations):
    seen, items = set(), []
    for citation in citations:
        key = (citation.paper_id, citation.chunk_id)
        if key not in seen:
            seen.add(key)
            items.append(citation)
    return items


def answer_export(message: ResearchConversationMessage) -> str:
    parts = ["# 研究回答", "", escape_untrusted(message.content), "", "## 引用证据", ""]
    for citation in unique_citations(message.citations):
        parts.extend([render_citation(citation), ""])
    if not message.citations:
        parts.append("本条回复没有论文引用，请勿视为已有文献支持的结论。")
    parts.append("\n引用位置可回查不等于每项结论均得到语义验证。")
    return "\n".join(parts)


def render_evidence_panel(st, message, index):
    citations = unique_citations(message.citations)
    if citations:
        papers = len({item.paper_id for item in citations})
        abstract = sum(item.page_number == 0 for item in citations)
        with st.expander(f"查看证据 · {papers} 篇论文 / {len(citations)} 条引用", expanded=False):
            st.caption(f"全文引用 {len(citations) - abstract} 条 · 摘要引用 {abstract} 条；展开原文核对结论。")
            for number, citation in enumerate(citations):
                st.markdown(render_citation(citation))
                if st.button("仅基于这篇继续提问", key=f"evidence_scope_{index}_{number}"):
                    st.session_state["research_scope_request"] = [citation.paper_id]
                    st.rerun()
    elif message.tool_name in {"evidence_qa", "research_plan"}:
        st.caption("本条回复没有可回查的论文引用。")
    st.download_button("导出回答与引用", answer_export(message), file_name="research-answer.md",
                       mime="text/markdown", key=f"export_answer_{index}")
