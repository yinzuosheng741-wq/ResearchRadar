"""Pure, dependency-free presentation helpers for the Streamlit demo."""

from __future__ import annotations

import html
import re
from typing import Any, Mapping


STATUS_FIELDS = (
    "metadata_total", "pdf_ready", "parsed", "profiled", "indexed", "abstract_only", "failed",
)
PROVIDERS = ("openalex", "unpaywall", "core", "crossref", "semantic_scholar")
_MARKDOWN = re.compile(r"([\\`*_{}\[\]()#+.!|>~-])")
_LINK_SCHEME = re.compile(r"(?i)(?:javascript|data|vbscript)\s*:")
_URL = re.compile(r"(?i)\bhttps?://[^\s<>()\]\[{}]+")
_WINDOWS_PATH = re.compile(r"(?i)\b[a-z]:[\\/](?:[^\s<>:\"|?*]+[\\/])*[^\s<>:\"|?*]*")
_POSIX_PATH = re.compile(r"(?<!\w)/(?:[^\s/]+/)+[^\s/]+")
_UNC_PATH = re.compile(r"\\\\[^\s\\/]+[\\/][^\s]+")
_SENSITIVE_LINE = re.compile(
    r"(?i)(?:traceback|[a-z0-9_]*(?:api[_ -]?key|access[_ -]?token)|secret|password|credential)\s*(?:[:=]|\b)[^\r\n]*"
)


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def escape_untrusted(value: Any) -> str:
    """Return literal Markdown-safe text with HTML and active schemes neutralized."""
    text = str(value)
    text = _SENSITIVE_LINE.sub("[已隐藏敏感内容]", text)
    text = _URL.sub("[已隐藏链接]", text)
    text = _WINDOWS_PATH.sub("[已隐藏路径]", text)
    text = _UNC_PATH.sub("[已隐藏路径]", text)
    text = _POSIX_PATH.sub("[已隐藏路径]", text)
    text = html.escape(text, quote=True)
    text = _LINK_SCHEME.sub("blocked-scheme:", text)
    return _MARKDOWN.sub(r"\\\1", text)


def render_citation(citation, *, abstract_only: bool | None = None) -> str:
    title = escape_untrusted(_value(citation, "title", "未命名论文"))
    page_number = _value(citation, "page_number", 0)
    evidence_label = "摘要证据" if abstract_only is True or page_number == 0 else f"第 {page_number} 页"
    quote = escape_untrusted(_value(citation, "quote", ""))
    return f"**{title}** · {evidence_label}\n\n> {quote}"


def _safe_nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed >= 0 else 0


def _safe_nonnegative_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if parsed >= 0 else 0.0


def render_agent_diagnostics(diagnostics) -> str:
    """Render only bounded, non-sensitive Agent runtime fields."""
    skill_id = _value(diagnostics, "skill_id", "unknown")
    if skill_id not in {"evidence_qa", "research_plan", "general_chat"}:
        skill_id = "unknown"
    route_mode = _value(diagnostics, "route_mode", "fallback")
    if route_mode not in {"model", "fallback"}:
        route_mode = "fallback"
    fallback = bool(_value(diagnostics, "fallback", False))
    route_reason = _value(diagnostics, "route_reason", "fallback_rule_general_chat")
    route_reason_labels = {
        "model_structured_route": "模型结构化路由",
        "fallback_rule_research_plan": "规则路由：研究路线",
        "fallback_rule_domain_question": "规则路由：领域证据问答",
        "fallback_rule_general_chat": "规则路由：普通对话",
    }
    skill_version = _value(diagnostics, "skill_version", "not_applicable")
    if not isinstance(skill_version, str) or len(skill_version) > 32:
        skill_version = "not_applicable"
    evidence_sufficient = bool(_value(diagnostics, "evidence_sufficient", False))
    return "\n".join(
        [
            "**运行诊断**",
            f"- Skill：{escape_untrusted(skill_id)}",
            f"- Skill 版本：{escape_untrusted(skill_version)}",
            f"- 路由：{escape_untrusted(route_mode)}",
            f"- 路由依据：{route_reason_labels.get(route_reason, route_reason_labels['fallback_rule_general_chat'])}",
            f"- 本地证据：{'充足' if evidence_sufficient else '不足'}",
            f"- 检索候选：{_safe_nonnegative_int(_value(diagnostics, 'retrieval_candidates', 0))}",
            f"- 证据块：{_safe_nonnegative_int(_value(diagnostics, 'evidence_chunks', 0))}",
            f"- 引用数：{_safe_nonnegative_int(_value(diagnostics, 'citation_count', 0))}",
            f"- Fallback：{'是' if fallback else '否'}",
            f"- 耗时：{_safe_nonnegative_float(_value(diagnostics, 'total_ms', 0.0)):.3f} ms",
        ]
    )


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def status_summary(stats) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in STATUS_FIELDS:
        count = _count(_value(stats, field))
        if count is not None:
            output[field] = count
    raw_providers = _value(stats, "providers", {})
    providers = {}
    if isinstance(raw_providers, Mapping):
        for name in PROVIDERS:
            count = _count(raw_providers.get(name))
            if count is not None:
                providers[name] = count
    if providers:
        output["providers"] = providers
    return output


def knowledge_metrics(snapshot: Mapping[str, Any]) -> dict[str, int]:
    return status_summary(_value(snapshot, "stats", {}))


def profile_coverage_summary(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    coverage = _value(snapshot, "profile_coverage", {})
    if not isinstance(coverage, Mapping):
        return {}
    output = {
        "profiled_papers": _count(coverage.get("profiled_papers")) or 0,
        "metadata_total": _count(coverage.get("metadata_total")) or 0,
        "coverage_ratio": f"{max(float(coverage.get('coverage_ratio', 0.0) or 0.0), 0.0):.2%}",
        "fulltext_profiled_papers": _count(coverage.get("fulltext_profiled_papers")) or 0,
        "fulltext_evidence_papers": _count(coverage.get("fulltext_evidence_papers")) or 0,
        "fulltext_coverage_ratio": f"{max(float(coverage.get('fulltext_coverage_ratio', 0.0) or 0.0), 0.0):.2%}",
    }
    return output


def distribution_series(snapshot: Mapping[str, Any], field: str) -> list[dict[str, Any]]:
    values = _value(snapshot, field, {})
    if not isinstance(values, Mapping):
        return []
    series = []
    for label, count in values.items():
        if isinstance(label, (str, int)) and isinstance(count, int) and count >= 0:
            series.append({"label": escape_untrusted(label), "count": count})
    return sorted(series, key=lambda item: str(item["label"]))


def paper_table_rows(papers) -> list[dict[str, str]]:
    columns = (
        "paper_id", "title", "year", "venue", "source", "doi", "status",
        "profiled", "evidence_chunks", "last_error",
    )
    rows = []
    for paper in papers:
        rows.append(
            {
                column: escape_untrusted(_value(paper, column, "未报告"))
                for column in columns
            }
        )
    return rows
