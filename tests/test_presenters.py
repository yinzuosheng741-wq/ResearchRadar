import importlib
import sys

import pytest

from domain.models import (
    AnswerCitation,
)
from web.presenters import (
    render_citation,
    status_summary,
)


def test_citation_renders_title_page_and_exact_quote_and_labels_abstract():
    citation = AnswerCitation(
        chunk_id="c1", paper_id="p1", title="Remote Sensing Study",
        page_number=3, quote="chlorophyll prediction improved",
    )
    rendered = render_citation(citation)
    assert "Remote Sensing Study" in rendered
    assert "第 3 页" in rendered
    assert "> chlorophyll prediction improved" in rendered

    abstract = citation.model_copy(update={"page_number": 0})
    assert "摘要证据" in render_citation(abstract)
    assert "第 0 页" not in render_citation(abstract)
    assert "摘要证据" in render_citation(citation, abstract_only=True)


def test_citation_escapes_markdown_and_html_in_untrusted_text():
    citation = AnswerCitation(
        chunk_id="c1", paper_id="p1",
        title="[x](javascript:alert(1)) <script>boom</script>",
        page_number=1,
        quote="**bold** <img src=x onerror=alert(1)>",
    )
    rendered = render_citation(citation)
    assert "<script>" not in rendered
    assert "<img" not in rendered
    assert "javascript:" not in rendered
    assert "\\[x\\]" in rendered
    assert "\\*\\*bold\\*\\*" in rendered


def test_status_summary_allows_only_known_integer_counts_and_providers():
    summary = status_summary({
        "metadata_total": 9, "pdf_ready": 4, "parsed": 3, "indexed": 2,
        "abstract_only": 1, "failed": 1,
        "providers": {"openalex": 7, "core": 2, "evil_url": "https://bad.test?key=secret"},
        "api_key": "secret", "traceback": "Traceback ...", "path": r"C:\\secret",
    })
    assert summary == {
        "metadata_total": 9, "pdf_ready": 4, "parsed": 3, "indexed": 2,
        "abstract_only": 1, "failed": 1,
        "providers": {"openalex": 7, "core": 2},
    }


def test_profile_coverage_summary_labels_scope_and_evidence_layers_safely():
    from web.presenters import profile_coverage_summary

    rendered = profile_coverage_summary(
        {
            "profile_coverage": {
                "profiled_papers": 11,
                "metadata_total": 100,
                "coverage_ratio": 11 / 100,
                "fulltext_profiled_papers": 10,
                "fulltext_evidence_papers": 96,
                "fulltext_coverage_ratio": 10 / 96,
            }
        }
    )

    assert rendered["profiled_papers"] == 11
    assert rendered["metadata_total"] == 100
    assert rendered["coverage_ratio"] == "11.00%"
    assert rendered["fulltext_coverage_ratio"] == "10.42%"


class FakeStreamlit:
    def __init__(self, *, buttons=None, text=""):
        self.buttons = set(buttons or [])
        self.text = text
        self.messages = []
        self.session_state = {}

    def _record(self, kind, value=""):
        self.messages.append((kind, str(value)))

    def title(self, value): self._record("title", value)
    def subheader(self, value): self._record("subheader", value)
    def info(self, value): self._record("info", value)
    def warning(self, value): self._record("warning", value)
    def error(self, value): self._record("error", value)
    def success(self, value): self._record("success", value)
    def write(self, value): self._record("write", value)
    def markdown(self, value): self._record("markdown", value)
    def caption(self, value): self._record("caption", value)
    def dataframe(self, value, **kwargs): self._record("dataframe", value)
    def bar_chart(self, value): self._record("bar_chart", value)
    def metric(self, label, value): self._record("metric", f"{label}:{value}")
    def button(self, label, **kwargs): return label in self.buttons
    def text_input(self, label, **kwargs): return self.text


class FakeStatus:
    def __init__(self, owner, label):
        self.owner = owner
        self.label = label

    def __enter__(self):
        self.owner._record("status", f"running:{self.label}")
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, **kwargs):
        self.owner._record("status", f"update:{kwargs.get('label')}:{kwargs.get('state')}")


class StatusStreamlit(FakeStreamlit):
    def status(self, label, **kwargs):
        return FakeStatus(self, label)


class EmptyServices:
    def __init__(self):
        self.calls = []

    def knowledge_snapshot(self):
        return {"stats": {"metadata_total": 0}, "years": {}, "sources": {},
                "recent_failures": [], "indexed_papers": []}

    def sync(self): self.calls.append(("sync",)); return {"status": "ok"}
    def cited_qa(self, question): self.calls.append(("qa", question)); return None
    def supplement_search(self, query): self.calls.append(("search", query)); return None


def _all_text(st):
    return "\n".join(value for _, value in st.messages)


def test_running_status_reports_running_and_completed_states():
    from web.app import _running_status

    streamlit = StatusStreamlit()
    with _running_status(streamlit, "正在执行测试", "测试已完成"):
        pass

    assert ("status", "running:正在执行测试") in streamlit.messages
    assert ("status", "update:测试已完成:complete") in streamlit.messages


def test_running_status_reports_failure_and_preserves_exception():
    from web.app import _running_status

    streamlit = StatusStreamlit()
    with pytest.raises(RuntimeError, match="boom"):
        with _running_status(streamlit, "正在执行测试", "测试已完成"):
            raise RuntimeError("boom")

    assert ("status", "update:操作失败:error") in streamlit.messages


def test_importing_web_app_without_credentials_constructs_no_services(monkeypatch):
    for name in ("OPENAI_API_KEY", "OPENALEX_API_KEY", "CORE_API_KEY", "UNPAYWALL_EMAIL"):
        monkeypatch.delenv(name, raising=False)
    sys.modules.pop("web.app", None)
    module = importlib.import_module("web.app")
    assert module._DEFAULT_SERVICES_CONSTRUCTED == 0


def test_core_pages_render_actionable_empty_states_without_service_actions():
    from web.app import render_knowledge_base_page, render_qa_page

    services = EmptyServices()
    streamlits = [FakeStreamlit() for _ in range(2)]
    render_knowledge_base_page(streamlits[0], services)
    render_qa_page(streamlits[1], services)
    assert "同步新增论文" in _all_text(streamlits[0])
    assert "输入问题" in _all_text(streamlits[1])
    assert services.calls == []


def test_actions_call_services_only_on_explicit_events_and_questions_are_not_cached():
    from domain.models import CitedAnswer
    from web.app import render_knowledge_base_page, render_qa_page

    services = EmptyServices()
    render_knowledge_base_page(FakeStreamlit(), services)
    assert services.calls == []
    render_knowledge_base_page(FakeStreamlit(buttons={"同步新增论文"}), services)
    assert services.calls == [("sync",)]

    services.cited_qa = lambda question: services.calls.append(("qa", question)) or CitedAnswer(
        answer_markdown="insufficient", evidence_sufficient=False,
        suggested_search_query=f"search {question}", citations=[],
    )
    render_qa_page(FakeStreamlit(text="question one"), services)
    render_qa_page(FakeStreamlit(text="question one", buttons={"提交问题"}), services)
    render_qa_page(FakeStreamlit(text="question two", buttons={"提交问题"}), services)
    assert [call for call in services.calls if call[0] == "qa"] == [
        ("qa", "question one"), ("qa", "question two")
    ]

    render_qa_page(
        FakeStreamlit(text="question three", buttons={"提交问题", "证据不足时补充检索"}),
        services,
    )
    assert services.calls[-2:] == [("qa", "question three"), ("search", "search question three")]


def test_insufficient_search_action_survives_streamlit_rerun_without_caching_answer():
    from domain.models import CitedAnswer
    from web.app import render_qa_page

    services = EmptyServices()
    services.cited_qa = lambda question: services.calls.append(("qa", question)) or CitedAnswer(
        answer_markdown="insufficient", evidence_sufficient=False,
        suggested_search_query=f"search {question}", citations=[],
    )
    streamlit = FakeStreamlit(text="question one", buttons={"提交问题"})
    render_qa_page(streamlit, services)
    streamlit.buttons = {"证据不足时补充检索"}
    render_qa_page(streamlit, services)
    assert services.calls == [("qa", "question one"), ("search", "search question one")]

    streamlit.text = "question two"
    streamlit.buttons = {"证据不足时补充检索"}
    render_qa_page(streamlit, services)
    assert services.calls == [("qa", "question one"), ("search", "search question one")]


def test_operational_exceptions_render_stable_code_not_sensitive_exception():
    from web.app import render_knowledge_base_page

    services = EmptyServices()
    def fail():
        raise RuntimeError(r"https://provider.test?token=secret C:\\private Traceback")
    services.sync = fail
    streamlit = FakeStreamlit(buttons={"同步新增论文"})
    render_knowledge_base_page(streamlit, services)
    text = _all_text(streamlit)
    assert "ui_sync_failed" in text
    assert "provider.test" not in text
    assert "private" not in text


def test_agent_errors_use_actionable_stable_codes():
    from web.app import _agent_error_code, _safe_error

    assert _agent_error_code(TimeoutError("upstream timeout")) == "ui_agent_timeout"
    assert _agent_error_code(RuntimeError("provider unavailable")) == "ui_agent_failed"

    streamlit = FakeStreamlit()
    _safe_error(streamlit, "ui_agent_timeout")
    assert "模型连接超时" in _all_text(streamlit)


def test_streamlit_services_load_cli_app_without_web_app_module_collision(monkeypatch):
    import web.app as web_app

    monkeypatch.setitem(sys.modules, "app", web_app)
    services = web_app.DefaultUiServices()
    cli = services._cli_services()
    assert cli.__class__.__name__ == "DefaultServices"


def test_recent_failures_render_only_allowlisted_stable_error_codes():
    from web.app import render_knowledge_base_page

    services = EmptyServices()
    services.knowledge_snapshot = lambda: {
        "stats": {"metadata_total": 2}, "years": {}, "sources": {},
        "indexed_papers": [], "recent_failures": [
            {"code": "pdf_parse_failed"},
            {"code": "OPENAI_API_KEY_secret"},
            {"code": r"C:\\private"},
        ],
    }
    streamlit = FakeStreamlit()
    render_knowledge_base_page(streamlit, services)
    text = _all_text(streamlit)
    # Failure details are intentionally shown only on the maintenance page,
    # not in the high-level knowledge overview.
    assert "pdf_parse_failed" not in text
    assert "OPENAI_API_KEY_secret" not in text
    assert "private" not in text


def test_knowledge_charts_do_not_forward_untrusted_source_aggregate_keys():
    from web.app import render_knowledge_base_page

    services = EmptyServices()
    services.knowledge_snapshot = lambda: {
        "stats": {"metadata_total": 2, "providers": {"openalex": 2}},
        "years": {2026: 2},
        "sources": {"https://bad.test?token=secret": 99},
        "indexed_papers": [], "recent_failures": [],
    }
    streamlit = FakeStreamlit()
    render_knowledge_base_page(streamlit, services)
    text = _all_text(streamlit)
    assert "openalex" in text
    assert "bad.test" not in text
    assert "secret" not in text


def test_sync_stable_failure_result_is_not_reported_as_success():
    from web.app import render_knowledge_base_page

    services = EmptyServices()
    services.sync = lambda: {"status": "sync_failed", "collected": 0, "ingested": 0}
    streamlit = FakeStreamlit(buttons={"同步新增论文"})
    render_knowledge_base_page(streamlit, services)
    text = _all_text(streamlit)
    assert "ui_sync_failed" in text
    assert "同步任务已完成" not in text


def test_resubmitting_same_question_clears_stale_insufficient_search_state():
    from domain.models import CitedAnswer
    from web.app import render_qa_page

    services = EmptyServices()
    answers = iter([
        CitedAnswer(answer_markdown="insufficient", evidence_sufficient=False,
                    suggested_search_query="stale search", citations=[]),
        CitedAnswer(answer_markdown="supported", evidence_sufficient=True, citations=[]),
    ])
    services.cited_qa = lambda question: next(answers)
    streamlit = FakeStreamlit(text="same", buttons={"提交问题"})
    render_qa_page(streamlit, services)
    render_qa_page(streamlit, services)
    streamlit.buttons = {"证据不足时补充检索"}
    render_qa_page(streamlit, services)
    assert services.calls == []


def test_presenter_redacts_urls_paths_credentials_and_traceback_text():
    citation = AnswerCitation(
        chunk_id="c", paper_id="p", title="https://provider.test/x?token=secret",
        page_number=1,
        quote=r"Traceback OPENAI_API_KEY=abc123 C:\\Users\\Admin\\private.pdf",
    )
    rendered = render_citation(citation)
    for forbidden in ("provider.test", "secret", "Traceback", "OPENAI_API_KEY", "Admin", "private.pdf"):
        assert forbidden not in rendered

def test_presenter_redacts_standalone_underscore_credentials_and_unc_paths():
    citation = AnswerCitation(
        chunk_id="c", paper_id="p", title="CORE_API_KEY=abc123",
        page_number=1,
        quote=r"OPENAI_API_KEY=TEST_PLACEHOLDER \\server\private\paper.pdf",
    )
    rendered = render_citation(citation)
    for forbidden in ("CORE_API_KEY", "TEST_PLACEHOLDER", "OPENAI_API_KEY", "server", "private", "paper.pdf"):
        assert forbidden not in rendered
