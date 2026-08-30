"""Credential-lazy native Streamlit demonstration interface."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import importlib.util
import socket
import sys
from pathlib import Path
from typing import Any, Protocol

from web.presenters import (
    distribution_series,
    escape_untrusted,
    knowledge_metrics,
    paper_table_rows,
    render_citation,
    render_agent_diagnostics,
    profile_coverage_summary,
    status_summary,
)
from utils.logger import logger


_DEFAULT_SERVICES_CONSTRUCTED = 0
_METRIC_LABELS = {
    "metadata_total": "元数据",
    "pdf_ready": "PDF 就绪",
    "parsed": "已解析",
    "profiled": "已画像",
    "indexed": "已索引",
    "abstract_only": "仅摘要",
    "failed": "失败",
}
WORKBENCH_NAV_ITEMS = (
    ("工作台", "科研助手"),
    ("工作台", "知识库"),
    ("工作台", "文献库"),
    ("工作台", "研究方案"),
    ("维护", "知识库维护"),
    ("维护", "任务与日志"),
    ("维护", "数据源设置"),
)
_FAILURE_CODES = frozenset({
    "full_text_resolution_failed", "ingestion_failed", "invalid_pdf",
    "metadata_enrichment_failed", "no_open_full_text", "pdf_download_failed",
    "pdf_parse_failed", "pdf_too_large", "profile_extraction_failed",
    "vector_index_failed",
})
_FAILURE_LABELS = {
    "full_text_resolution_failed": "全文地址解析失败",
    "ingestion_failed": "文献摄入失败",
    "invalid_pdf": "下载内容不是有效 PDF",
    "metadata_enrichment_failed": "元数据校正失败",
    "no_open_full_text": "没有可用的开放全文",
    "pdf_download_failed": "PDF 下载失败",
    "pdf_parse_failed": "PDF 解析失败",
    "pdf_too_large": "PDF 超过大小限制",
    "profile_extraction_failed": "论文画像生成失败",
    "vector_index_failed": "向量索引失败",
}
_PENDING_QA_KEY = "_pending_insufficient_qa"
_RESEARCH_AGENT_STATE_KEY = "research_agent_conversation"
_RESEARCH_AGENT_SEARCH_KEY = "research_agent_pending_search"


@contextmanager
def _running_status(st, running_label: str, complete_label: str):
    """Show a consistent running/completed state for potentially slow UI work."""
    status = None
    if hasattr(st, "status"):
        try:
            status = st.status(running_label, expanded=True)
        except TypeError:
            status = st.status(running_label)
    elif hasattr(st, "spinner"):
        status = st.spinner(running_label)
    else:
        status = nullcontext()
    try:
        with status:
            yield
    except Exception:
        if hasattr(status, "update"):
            status.update(label="操作失败", state="error", expanded=False)
        raise
    else:
        if hasattr(status, "update"):
            status.update(label=complete_label, state="complete", expanded=False)


class UiServices(Protocol):
    def knowledge_snapshot(self) -> dict[str, Any]: ...
    def knowledge_audit(self) -> dict[str, Any]: ...
    def sync(self): ...
    def cited_qa(self, question: str): ...
    def research_chat(self, message: str, conversation=None): ...
    def supplement_search(self, query: str): ...
    def profile(self, *, retry_failed: bool = False): ...
    def rebuild_index(self): ...
    def provider_health(self): ...


def _safe_error(st, code: str) -> None:
    messages = {
        "ui_agent_timeout": "模型连接超时，回答未完成；请稍后重试或缩短问题。",
        "ui_agent_failed": "科研助手暂时不可用，请检查模型配置后重试。",
        "ui_provider_health_failed": "数据源健康检查失败，请检查网络或配置后重试。",
    }
    st.error(messages.get(code, f"操作未完成（{code}），请检查本地配置后重试。"))


def _agent_error_code(error: Exception) -> str:
    """Map transient upstream failures to a useful, secret-safe UI state."""
    message = f"{type(error).__name__} {error}".lower()
    if isinstance(error, (TimeoutError, socket.timeout)) or any(
        marker in message for marker in ("timeout", "timed out", "stream disconnected", "upstream request failed")
    ):
        return "ui_agent_timeout"
    return "ui_agent_failed"


def _snapshot(services: UiServices) -> dict[str, Any]:
    try:
        value = services.knowledge_snapshot()
    except Exception:
        return {"stats": {}, "years": {}, "sources": {}, "recent_failures": [], "indexed_papers": []}
    return value if isinstance(value, dict) else {}


def render_sidebar(st) -> str:
    grouped: dict[str, list[str]] = {}
    for group, label in WORKBENCH_NAV_ITEMS:
        grouped.setdefault(group, []).append(label)
    active = st.session_state.get("research_workbench_module", "科研助手")
    available_labels = {label for _group, label in WORKBENCH_NAV_ITEMS}
    if active not in available_labels:
        active = "科研助手"
        st.session_state["research_workbench_module"] = active
    style = """
    <style>
    [data-testid="stSidebar"] {
      background: #fbfcfe;
      border-right: 1px solid #e6ebf1;
    }
    [data-testid="stSidebar"] > div:first-child {
      padding: 26px 13px 28px 18px;
    }
    [data-testid="stSidebar"] .workbench-brand {
      color: #1e3147;
      font-size: 17px;
      font-weight: 700;
      letter-spacing: .01em;
      margin: 0 0 30px 3px;
    }
    [data-testid="stSidebar"] .workbench-section {
      color: #7890a8;
      font-size: 13px;
      font-weight: 600;
      margin: 22px 0 8px 4px;
    }
    [data-testid="stSidebar"] .workbench-divider {
      border-top: 1px solid #e7edf3;
      margin: 22px 0 7px;
    }
    [data-testid="stSidebar"] button[kind="secondary"] {
      border: 0;
      border-radius: 6px;
      color: #26394f;
      font-size: 14px;
      line-height: 1.35;
      margin: 2px 0;
      min-height: 38px;
      padding: 9px 10px 9px 11px;
      text-align: left;
      justify-content: flex-start;
      transition: background .15s ease, color .15s ease;
    }
    [data-testid="stSidebar"] button[kind="secondary"]:hover {
      background: #f1f6fb;
      color: #176bb0;
    }
    [data-testid="stSidebar"] button[data-active="true"] {
      background: #e6f1ff;
      color: #176bb0;
      font-weight: 600;
    }
    [data-testid="stSidebar"] .workbench-nav-item {
      align-items: center;
      background: #e6f1ff;
      border-radius: 6px;
      color: #176bb0;
      display: flex;
      font-size: 14px;
      font-weight: 600;
      line-height: 1.35;
      margin: 2px 0;
      min-height: 38px;
      padding: 9px 10px 9px 11px;
    }
    </style>
    """
    def sidebar_markdown(value: str) -> None:
        try:
            st.sidebar.markdown(value, unsafe_allow_html=True)
        except TypeError:
            st.sidebar.markdown(value)

    try:
        sidebar_markdown(style)
        sidebar_markdown('<div class="workbench-brand">科研工作台</div>')
        for index, (group, group_labels) in enumerate(grouped.items()):
            if index:
                sidebar_markdown('<div class="workbench-divider"></div>')
            sidebar_markdown(f'<div class="workbench-section">{group}</div>')
            for label in group_labels:
                display = label
                if label == active:
                    sidebar_markdown(f'<div class="workbench-nav-item">{display}</div>')
                    continue
                clicked = st.sidebar.button(
                    display,
                    key=f"research_nav_{label}",
                    width="stretch",
                )
                if clicked:
                    active = label
                    st.session_state["research_workbench_module"] = label
                    if hasattr(st, "rerun"):
                        st.rerun()
    except TypeError:
        # Keeps the presenter tests' tiny fake Streamlit implementation usable.
        st.sidebar.markdown(style)
    return active


def _chart_data(snapshot: dict[str, Any], field: str) -> dict[str, int]:
    return {
        item["label"]: item["count"]
        for item in distribution_series(snapshot, field)
        if str(item["label"]).strip().lower() not in {"未分类", "未报告", "unknown", "none", ""}
    }


def render_knowledge_overview_page(st, services: UiServices) -> None:
    st.title("知识库")
    st.caption("查看文献规模、研究结构、证据覆盖和当前知识库状态")
    snapshot = _snapshot(services)
    summary = knowledge_metrics(snapshot)
    venues = _chart_data(snapshot, "venues")
    metric_values = (
        ("论文总数", summary.get("metadata_total", 0)),
        ("已画像", summary.get("profiled", 0)),
        ("文本块", snapshot.get("chunks_total", 0)),
        ("仅摘要", summary.get("abstract_only", 0)),
        ("已索引", summary.get("indexed", 0)),
    )
    if hasattr(st, "columns"):
        metric_columns = st.columns(len(metric_values))
        for column, (label, value) in zip(metric_columns, metric_values):
            with column:
                st.metric(label, value)
    else:
        for label, value in metric_values:
            st.metric(label, value)
    total = summary.get("metadata_total", 0)
    if not total:
        st.info("知识库尚无论文。请前往“知识库维护”开始构建个人文献库。")
    elif summary.get("abstract_only", 0):
        st.warning(
            f"当前有 {summary.get('abstract_only', 0)} 篇论文仅保存摘要，不能作为完整全文证据；"
            "建议在知识库维护中优先补充开放全文。"
        )
    years = _chart_data(snapshot, "years")
    if hasattr(st, "columns"):
        chart_columns = st.columns(2)
        with chart_columns[0]:
            st.subheader("发表年份分布")
            if years:
                st.bar_chart(years)
            else:
                st.info("暂无年份数据")
        with chart_columns[1]:
            st.subheader("期刊 / 会议分布")
            if venues:
                st.dataframe(
                    [{"期刊 / 会议": label, "论文数": count} for label, count in sorted(venues.items(), key=lambda item: (-item[1], str(item[0])))[:8]],
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.info("暂无期刊数据")
    else:
        if years:
            st.subheader("发表年份分布")
            st.bar_chart(years)
        if venues:
            st.subheader("期刊 / 会议分布")
            st.dataframe([{"期刊 / 会议": label, "论文数": count} for label, count in venues.items()])

    st.subheader("数据覆盖")
    st.caption(
        f"当前目录包含 {summary.get('metadata_total', 0)} 篇元数据、"
        f"{summary.get('indexed', 0)} 篇已索引论文和 {snapshot.get('chunks_total', 0)} 个证据块。"
    )

    st.subheader("最近入库论文")
    papers = snapshot.get("papers", [])
    recent = [
        {"标题": paper.get("title", "未命名论文"), "年份": paper.get("year") or "未报告", "状态": paper.get("status", "未报告")}
        for paper in papers[:6]
        if isinstance(paper, dict)
    ]
    if recent:
        st.dataframe(recent, hide_index=True, width="stretch")
    else:
        st.info("暂无入库论文")
    sources = summary.get("providers", {})
    if sources:
        st.subheader("数据来源")
        st.bar_chart(dict(sorted(sources.items())))
    st.caption("同步新增论文可从此处快速触发，也可前往“知识库维护”查看任务状态")
    if st.button("同步新增论文", key="overview_sync"):
        try:
            with _running_status(st, "正在同步新增论文...", "同步任务已完成"):
                result = services.sync()
            if isinstance(result, dict) and result.get("status") == "ok":
                st.success("同步任务已完成。")
            else:
                _safe_error(st, "ui_sync_failed")
        except Exception:
            _safe_error(st, "ui_sync_failed")


def render_knowledge_base_page(st, services: UiServices) -> None:
    """Backward-compatible alias for the workbench knowledge page."""
    render_knowledge_overview_page(st, services)


def render_literature_library_page(st, services: UiServices) -> None:
    st.title("文献库")
    st.caption("浏览已入库论文，并按标题、DOI 或摄入状态筛选")
    snapshot = _snapshot(services)
    papers = snapshot.get("papers", snapshot.get("indexed_papers", []))
    query = st.text_input("按标题或 DOI 筛选", key="library_query").strip().lower()
    status = st.selectbox(
        "摄入状态",
        ["全部", "indexed", "profiled", "abstract_only", "failed"],
        key="library_status",
    )
    filtered = []
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        haystack = " ".join(str(paper.get(key, "")) for key in ("title", "doi")).lower()
        if query and query not in haystack:
            continue
        if status != "全部" and paper.get("status") != status:
            continue
        filtered.append(paper)
    st.caption(f"匹配 {len(filtered)} 篇论文")
    if filtered:
        st.dataframe(paper_table_rows(filtered), width="stretch")
    else:
        st.info("暂无符合条件的论文。请调整筛选条件或先前往知识库维护同步文献。")


def render_import_sync_page(st, services: UiServices) -> None:
    st.title("导入与同步")
    st.info("使用现有 seed/sync 流程构建本地文献库；全文只处理合法开放获取版本。")
    if st.button("同步新增论文", key="workbench_sync"):
        try:
            with _running_status(st, "正在同步新增论文...", "同步任务已完成"):
                result = services.sync()
            if isinstance(result, dict) and result.get("status") == "ok":
                st.success("同步任务已完成。")
            else:
                _safe_error(st, "ui_sync_failed")
        except Exception:
            _safe_error(st, "ui_sync_failed")
    doi = st.text_input("手动输入 DOI（仅记录入口）", key="manual_doi").strip()
    if st.button("加入待处理列表", key="add_doi"):
        if doi:
            st.info("手动 DOI 入口将在下一次受控摄入中处理。")
        else:
            st.warning("请输入 DOI。")
    st.caption("本地 PDF 上传沿用同一摄入流水线，运行时文件保存在 data/papers。")


def render_research_plan_page(st, services: UiServices) -> None:
    from domain.models import ResearchConversationState

    st.title("研究方案")
    st.caption("根据本地论文证据生成一个可验证的起步路线")
    topic = st.text_input(
        "研究问题",
        placeholder="例如：用 Sentinel-2 预测湖泊叶绿素 a，应该如何设计基线实验？",
        key="research_plan_topic",
    ).strip()
    if not st.button("生成研究路线", key="generate_research_plan"):
        return
    if not topic:
        st.warning("请输入研究问题。")
        return
    if len(topic) > 4000:
        st.warning("研究问题不能超过 4000 个字符。")
        return
    try:
        with _running_status(st, "正在检索文献并生成研究路线...", "研究路线已生成"):
            turn = services.research_chat(topic, ResearchConversationState())
    except Exception as exc:
        _safe_error(st, _agent_error_code(exc))
        return
    st.markdown(escape_untrusted(turn.reply.content))
    for citation in turn.reply.citations:
        st.markdown(render_citation(citation))
    if not turn.reply.evidence_sufficient:
        st.warning("当前证据不足，建议先补充相关开放论文。")
        if turn.reply.suggested_search_query:
            st.caption(f"建议检索：{escape_untrusted(turn.reply.suggested_search_query)}")


def render_knowledge_maintenance_page(st, services: UiServices) -> None:
    st.title("知识库维护")
    st.caption("管理本地文献摄入、论文画像与向量索引。")
    snapshot = _snapshot(services)
    summary = knowledge_metrics(snapshot)
    try:
        with _running_status(st, "正在检查知识库...", "知识库检查已完成"):
            audit = services.knowledge_audit()
    except Exception:
        audit = {}
    audit = audit if isinstance(audit, dict) else {}
    st.write(
        {
            "文献元数据": summary.get("metadata_total", 0),
            "有摘要": audit.get("papers_with_abstract", 0),
            "有文本证据": audit.get("papers_with_chunks", 0),
            "文本块": audit.get("chunks_total", 0),
            "已画像论文": audit.get("profiled_papers", summary.get("profiled", 0)),
            "向量文档": audit.get("vector_indexed", 0),
            "仅摘要": audit.get("abstract_only_papers", summary.get("abstract_only", 0)),
            "最近失败": len(snapshot.get("recent_failures", [])),
        }
    )
    vector_index = audit.get("vector_index", {})
    missing = vector_index.get("missing_chunk_ids", []) if isinstance(vector_index, dict) else []
    orphan = vector_index.get("orphan_vector_ids", []) if isinstance(vector_index, dict) else []
    if missing or orphan:
        st.warning(f"向量索引存在缺口：缺少 {len(missing)} 个文本块，发现 {len(orphan)} 个孤立向量。")
    elif audit:
        st.success("知识库体检通过：SQLite 文本证据与向量文档数量一致。")
    layers = audit.get("evidence_layers", {}) if isinstance(audit, dict) else {}
    coverage = profile_coverage_summary(audit)
    if layers:
        st.subheader("证据层级")
        st.write(
            {
                "metadata catalog": layers.get("metadata_catalog", 0),
                "abstract evidence": layers.get("abstract_evidence", 0),
                "page-addressable full-text": layers.get("page_addressable_fulltext", 0),
                "abstract-only": layers.get("abstract_only", 0),
            }
        )
    if coverage:
        st.subheader("画像覆盖率")
        st.write(coverage)
    actions = st.columns(3)
    with actions[0]:
        sync_clicked = st.button("同步文献", key="maintenance_sync")
    with actions[1]:
        profile_clicked = st.button("生成论文画像", key="maintenance_profile")
    with actions[2]:
        rebuild_clicked = st.button("重建向量索引", key="maintenance_rebuild")
    try:
        if sync_clicked:
            with _running_status(st, "正在同步文献...", "同步任务已提交"):
                services.sync()
            st.success("同步任务已提交。")
        elif profile_clicked:
            with _running_status(st, "正在生成论文画像...", "论文画像已生成"):
                result = services.profile()
            st.success(f"已完成 {result.get('profiled', 0)} 篇论文画像。")
        elif rebuild_clicked:
            with _running_status(st, "正在重建向量索引...", "向量索引已重建"):
                count = services.rebuild_index()
            st.success(f"已重建 {count} 条向量。")
    except Exception:
        _safe_error(st, "ui_maintenance_failed")
    failures = snapshot.get("recent_failures", [])
    if failures:
        st.subheader("最近失败记录")
        grouped: dict[str, dict[str, Any]] = {}
        for failure in failures:
            if not isinstance(failure, dict):
                continue
            code = failure.get("code")
            if code not in _FAILURE_CODES:
                continue
            item = grouped.setdefault(code, {"count": 0, "titles": []})
            item["count"] += 1
            title = str(failure.get("title", "未命名论文")).strip()
            if title and title not in item["titles"] and len(item["titles"]) < 3:
                item["titles"].append(title)
        rows = [
            {
                "失败类型": _FAILURE_LABELS.get(code, "文献处理失败"),
                "数量": item["count"],
                "示例论文": "；".join(escape_untrusted(title) for title in item["titles"]) or "未报告",
            }
            for code, item in sorted(grouped.items())
        ]
        if rows:
            st.dataframe(rows, width="stretch")
        else:
            st.info("暂无可展示的失败记录。")


def render_task_log_page(st, services: UiServices) -> None:
    st.title("任务与日志")
    snapshot = _snapshot(services)
    tasks = snapshot.get("tasks", [])
    if not tasks:
        st.info("暂无任务记录。执行 seed 或同步后，运行状态会显示在这里。")
        return
    rows = [
        {
            "任务": escape_untrusted(item.get("kind", "未报告")),
            "状态": escape_untrusted(item.get("status", "未报告")),
            "开始时间": escape_untrusted(item.get("started_at", "未报告")),
            "结束时间": escape_untrusted(item.get("finished_at", "未完成")),
            "发现": item.get("discovered", 0),
            "下载": item.get("downloaded", 0),
            "索引": item.get("indexed", 0),
        }
        for item in tasks
        if isinstance(item, dict)
    ]
    if rows:
        st.dataframe(rows, width="stretch")
    else:
        st.info("暂无任务记录。")


def render_data_source_page(st, services: UiServices) -> None:
    st.title("数据源设置")
    snapshot = _snapshot(services)
    health = snapshot.get("provider_health", {})
    if st.button("运行数据源健康检查", key="provider_health_check"):
        try:
            with _running_status(st, "正在检查数据源...", "数据源检查已完成"):
                health = services.provider_health()
        except Exception:
            _safe_error(st, "ui_provider_health_failed")
            return
    if not health:
        st.info("尚未执行数据源健康检查。点击上方按钮开始检查。")
        return
    st.dataframe(health, width="stretch")


def render_qa_page(st, services: UiServices) -> None:
    st.title("证据问答")
    st.info("输入问题并提交；回答只引用已索引的论文证据。")
    question = st.text_input("输入问题")
    submitted = st.button("提交问题")
    pending = st.session_state.get(_PENDING_QA_KEY)
    if not submitted:
        if not isinstance(pending, dict) or pending.get("question") != question.strip():
            st.session_state.pop(_PENDING_QA_KEY, None)
            return
        st.warning("当前证据不足，可补充检索相关论文。")
        st.caption("证据不足时补充检索")
        if st.button("证据不足时补充检索"):
            try:
                with _running_status(st, "正在补充检索...", "补充检索已完成"):
                    services.supplement_search(pending["query"])
                st.session_state.pop(_PENDING_QA_KEY, None)
                st.success("补充检索已完成。")
            except Exception:
                _safe_error(st, "ui_search_failed")
        return
    if not question.strip():
        st.warning("请输入研究问题。")
        return
    st.session_state.pop(_PENDING_QA_KEY, None)
    try:
        with _running_status(st, "正在检索证据并生成回答...", "回答已生成"):
            answer = services.cited_qa(question.strip())
    except Exception:
        _safe_error(st, "ui_qa_failed")
        return
    if answer is None:
        st.info("当前没有足够的索引证据，请先同步论文。")
        return
    st.markdown(escape_untrusted(getattr(answer, "answer_markdown", "")))
    for citation in getattr(answer, "citations", []):
        st.markdown(render_citation(citation))
    if not getattr(answer, "evidence_sufficient", False):
        query = getattr(answer, "suggested_search_query", None)
        st.warning("当前证据不足，可补充检索相关论文。")
        st.caption("证据不足时补充检索")
        if query:
            st.session_state[_PENDING_QA_KEY] = {"question": question.strip(), "query": query}
        if query and st.button("证据不足时补充检索"):
            try:
                with _running_status(st, "正在补充检索...", "补充检索已完成"):
                    services.supplement_search(query)
                st.success("补充检索已完成。")
            except Exception:
                _safe_error(st, "ui_search_failed")


def render_research_agent_page(st, services: UiServices) -> None:
    from domain.models import ResearchConversationState

    st.title("科研 Agent")
    st.caption("基于本地论文知识库的证据化研究助手")
    controls = st.columns([1, 4])
    with controls[0]:
        if st.button("清空对话", key="clear_research_agent"):
            st.session_state.pop(_RESEARCH_AGENT_STATE_KEY, None)
            st.session_state.pop(_RESEARCH_AGENT_SEARCH_KEY, None)
            st.rerun()

    conversation = ResearchConversationState.model_validate(
        st.session_state.get(_RESEARCH_AGENT_STATE_KEY) or {}
    )
    if conversation.scope != type(conversation.scope)():
        with st.expander("当前研究范围"):
            scope = {
                key: value
                for key, value in conversation.scope.model_dump().items()
                if value
            }
            st.json(scope)
    if not conversation.messages:
        st.markdown(
            "<div class='agent-welcome'><div class='agent-welcome-title'>从一个研究问题开始</div>"
            "<div class='agent-welcome-copy'>Agent 会先判断任务，再调用证据问答或研究路线技能。</div>"
            "<div class='agent-suggestion'>例如：我想用 Sentinel-2 研究湖泊叶绿素 a 预测，应该从哪里开始？</div></div>",
            unsafe_allow_html=True,
        )
    tool_labels = {
        "evidence_qa": "证据问答",
        "research_plan": "研究路线",
        "general_chat": "普通对话（未调用本地知识库）",
    }
    for message in conversation.messages:
        with st.chat_message(message.role):
            st.markdown(escape_untrusted(message.content))
            if message.tool_name:
                st.caption(f"Agent 工具：{tool_labels.get(message.tool_name, message.tool_name)}")
            if message.diagnostics is not None:
                with st.expander("运行诊断", expanded=False):
                    st.markdown(render_agent_diagnostics(message.diagnostics))
            for citation in message.citations:
                st.markdown(render_citation(citation))

    pending_query = st.session_state.get(_RESEARCH_AGENT_SEARCH_KEY)
    if isinstance(pending_query, str) and pending_query:
        st.warning("当前证据不足，可先补充检索相关开放论文。")
        if st.button("补充检索", key="research_agent_supplement"):
            try:
                with _running_status(st, "正在补充检索...", "补充检索已完成"):
                    services.supplement_search(pending_query)
                st.session_state.pop(_RESEARCH_AGENT_SEARCH_KEY, None)
                st.success("补充检索已完成。")
            except Exception:
                _safe_error(st, "ui_search_failed")

    prompt = st.chat_input("输入科研问题", key="research_agent_input")
    if not prompt:
        return
    # Render the submitted question before the model/tool call so the chat feels immediate.
    with st.chat_message("user"):
        st.markdown(escape_untrusted(prompt))
    try:
        with _running_status(st, "正在处理你的问题", "回答已生成"):
            turn = services.research_chat(prompt, conversation)
    except ValueError:
        _safe_error(st, "ui_agent_message_invalid")
        return
    except Exception as exc:
        code = _agent_error_code(exc)
        logger.exception("research agent request failed code=%s error_type=%s", code, type(exc).__name__)
        _safe_error(st, code)
        return
    st.session_state[_RESEARCH_AGENT_STATE_KEY] = turn.state.model_dump(mode="json")
    if not turn.reply.evidence_sufficient and turn.reply.suggested_search_query:
        st.session_state[_RESEARCH_AGENT_SEARCH_KEY] = turn.reply.suggested_search_query
    else:
        st.session_state.pop(_RESEARCH_AGENT_SEARCH_KEY, None)
    st.rerun()


class DefaultUiServices:
    """Construct local and external dependencies only inside called operations."""

    def __init__(self) -> None:
        global _DEFAULT_SERVICES_CONSTRUCTED
        _DEFAULT_SERVICES_CONSTRUCTED += 1

    @staticmethod
    def _cli_services():
        # Streamlit executes this file as ``app``; importing ``app`` directly
        # would resolve back to ``web/app.py`` instead of the CLI entry point.
        module_name = "_intel_agent_cli_app"
        module = sys.modules.get(module_name)
        if module is None:
            path = Path(__file__).resolve().parents[1] / "app.py"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError("cli_app_loader_unavailable")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        return module.DefaultServices()

    def knowledge_snapshot(self) -> dict[str, Any]:
        services = self._cli_services()
        database = services._database()
        snapshot = database.knowledge_statistics()
        snapshot["sources"] = snapshot.get("providers", {})
        profiled_ids = set(snapshot.get("profiled_paper_ids", []))
        chunk_counts = snapshot.get("paper_chunk_counts", {})
        snapshot["papers"] = [
            {
                **paper.model_dump(mode="json"),
                "profiled": paper.paper_id in profiled_ids,
                "evidence_chunks": int(chunk_counts.get(paper.paper_id, 0)),
            }
            for paper in database.list_papers(limit=1000)
        ]
        snapshot["indexed_papers"] = [
            {
                **paper.model_dump(mode="json"),
                "profiled": paper.paper_id in profiled_ids,
                "evidence_chunks": int(chunk_counts.get(paper.paper_id, 0)),
            }
            for paper in database.list_papers(status="indexed", limit=1000)
        ]
        snapshot["tasks"] = database.recent_sync_runs()
        return snapshot

    def knowledge_audit(self) -> dict[str, Any]:
        services = self._cli_services()
        return services.knowledge_audit()

    def sync(self): return self._cli_services().sync()
    def cited_qa(self, question: str): return self._cli_services().cited_qa(question)
    def research_chat(self, message: str, conversation=None): return self._cli_services().research_chat(message, conversation)
    def supplement_search(self, query: str): return self._cli_services().collect_papers(type("Args", (), {"queries": query, "provider": "openalex", "max_results": 20, "include_pdf": True})())
    def profile(self, *, retry_failed: bool = False): return self._cli_services().profile(retry_failed=retry_failed)
    def rebuild_index(self): return self._cli_services().rebuild_index()
    def provider_health(self): return self._cli_services().provider_health()


def run_app(st=None, services: UiServices | None = None) -> None:
    if st is None:
        import streamlit as st
        st.markdown(
            """
            <style>
            .block-container { max-width: 1120px; padding-top: 3.2rem; padding-bottom: 7rem; }
            [data-testid="stAppViewContainer"] { background: #ffffff; }
            [data-testid="stHeader"] { background: transparent; }
            h1 { color: #1d3550; letter-spacing: -.02em; }
            [data-testid="stChatMessage"] { border: 1px solid #e9eef4; border-radius: 8px; padding: 1rem 1.1rem; margin-bottom: .65rem; }
            [data-testid="stChatInput"] { border-color: #d8e2ec; box-shadow: 0 6px 22px rgba(31, 61, 91, .08); }
            .agent-welcome { border: 1px solid #e5ecf3; border-radius: 8px; padding: 22px 24px; background: #fbfdff; margin: 1.2rem 0 1.4rem; }
            .agent-welcome-title { color: #23425f; font-size: 1.12rem; font-weight: 650; margin-bottom: .4rem; }
            .agent-welcome-copy { color: #6b7f93; font-size: .92rem; margin-bottom: .95rem; }
            .agent-suggestion { color: #246da8; background: #edf6ff; border-radius: 6px; padding: .7rem .85rem; font-size: .9rem; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    services = services or DefaultUiServices()
    pages = {
        "科研助手": render_research_agent_page,
        "知识库": render_knowledge_overview_page,
        "文献库": render_literature_library_page,
        "研究方案": render_research_plan_page,
        "知识库维护": render_knowledge_maintenance_page,
        "任务与日志": render_task_log_page,
        "数据源设置": render_data_source_page,
    }
    selected = render_sidebar(st)
    pages[selected](st, services)


if __name__ == "__main__":
    run_app()
