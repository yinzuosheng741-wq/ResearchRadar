"""Deterministic retrieval and citation evaluation with safe atomic reports."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
from typing import Any, Callable

from evaluation.dataset import EvaluationQuestion, load_answer_rows, load_questions
from evaluation.metrics import (
    citation_precision,
    deterministic_mean,
    evidence_group_recall_at_k,
    evidence_coverage,
    recall_at_k,
    reciprocal_rank,
    unsupported_claim_rate,
    answer_level_metrics,
)


class EvaluationError(RuntimeError):
    """Stable evaluator/report boundary error."""


@dataclass(frozen=True)
class EvaluationResult:
    json_path: Path
    markdown_path: Path
    metrics: dict[str, Any]
    acceptance: dict[str, Any]

    def model_dump(self, *, mode: str = "json") -> dict[str, Any]:
        return {
            "status": "ok",
            "json_report": self.json_path.name,
            "markdown_report": self.markdown_path.name,
            "metrics": self.metrics,
            "acceptance": self.acceptance,
        }


def run_answer_evaluation(
    dataset_path: str | Path,
    *,
    reports_dir: str | Path = Path("data/reports/evaluation"),
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Evaluate manually reviewed answer/citation rows without an LLM call."""
    rows = load_answer_rows(dataset_path)
    timestamp, stamp = _timestamp((now or (lambda: datetime.now(timezone.utc)))())
    payload = {
        "timestamp_utc": timestamp,
        "evaluation_scope": "answer_level_citation",
        "dataset_summary": {"question_count": len(rows)},
        "metrics": answer_level_metrics(rows),
        "questions": rows,
    }
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    json_path, markdown_path = _report_paths(reports_path, f"answers-{stamp}")
    _atomic_write(json_path, json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    lines = [
        "# Answer Evaluation Summary",
        "",
        f"- Timestamp (UTC): {timestamp}",
        f"- Dataset questions: {len(rows)}",
        f"- Citation precision: {payload['metrics']['citation_precision']:.6f}",
        f"- Evidence coverage: {payload['metrics']['evidence_coverage']:.6f}",
        f"- Unsupported claim rate: {payload['metrics']['unsupported_claim_rate']:.6f}",
        "",
        "注意：该结果来自人工整理的答案/引用样本，不代表领域通用线上准确率。",
    ]
    _atomic_write(markdown_path, "\n".join(lines) + "\n")
    return {
        "status": "ok",
        "json_report": json_path.name,
        "markdown_report": markdown_path.name,
        "metrics": payload["metrics"],
    }


def _fingerprint(questions: list[EvaluationQuestion]) -> str:
    canonical = [
        {
            "question_id": item.question_id,
            "question": item.question,
            "category": item.category,
            "relevant_paper_ids": item.relevant_paper_ids,
            "relevant_chunk_ids": item.relevant_chunk_ids,
        }
        for item in questions
    ]
    return hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _timestamp(value: datetime) -> tuple[str, str]:
    utc = value.astimezone(timezone.utc)
    return utc.isoformat().replace("+00:00", "Z"), utc.strftime("%Y%m%dT%H%M%SZ")


def _safe_chunk_ids(items: object) -> list[str]:
    if not isinstance(items, list):
        return []
    return [
        item.chunk_id
        for item in items
        if isinstance(getattr(item, "chunk_id", None), str)
    ]


def _safe_retrieval_trace(retriever: object) -> dict[str, Any] | None:
    trace = getattr(retriever, "last_trace", None)
    if trace is None or not is_dataclass(trace):
        return None
    values = asdict(trace)
    return {
        "query": str(values.get("query", ""))[:4000],
        "keyword_candidates": int(values.get("keyword_candidates", 0)),
        "vector_candidates": int(values.get("vector_candidates", 0)),
        "fused_candidates": int(values.get("fused_candidates", 0)),
        "selected_count": int(values.get("selected_count", 0)),
        "selected_chunk_ids": [str(item) for item in values.get("selected_chunk_ids", [])[:8]],
        "selected_paper_ids": [str(item) for item in values.get("selected_paper_ids", [])[:8]],
        "latency_ms": round(float(values.get("latency_ms", 0.0)), 3),
        "query_variants": [str(item) for item in values.get("query_variants", [])[:6]],
        "fallback_used": bool(values.get("fallback_used", False)),
        "retrieval_confidence": round(float(values.get("retrieval_confidence", 0.0)), 3),
        "stage1_keyword_candidates": int(values.get("stage1_keyword_candidates", 0)),
        "stage1_vector_candidates": int(values.get("stage1_vector_candidates", 0)),
        "candidate_paper_ids": [str(item) for item in values.get("candidate_paper_ids", [])[:16]],
        "stage2_candidates": int(values.get("stage2_candidates", 0)),
    }


def _safe_qa_trace(qa: object) -> dict[str, Any] | None:
    trace = getattr(qa, "last_trace", None)
    if trace is None or not is_dataclass(trace):
        return None
    values = asdict(trace)
    return {
        "status": str(values.get("status", "")),
        "retrieval_ms": round(float(values.get("retrieval_ms", 0.0)), 3),
        "model_ms": round(float(values.get("model_ms", 0.0)), 3),
        "retrieved_chunks": int(values.get("retrieved_chunks", 0)),
        "canonical_chunks": int(values.get("canonical_chunks", 0)),
        "citation_count": int(values.get("citation_count", 0)),
        "original_query": str(values.get("original_query", ""))[:1000],
        "query_variants": [str(item) for item in values.get("query_variants", [])[:6]],
        "intent_confidence": round(float(values.get("intent_confidence", 0.0)), 3),
        "retrieval_confidence": round(float(values.get("retrieval_confidence", 0.0)), 3),
        "evidence_confidence": round(float(values.get("evidence_confidence", 0.0)), 3),
        "answerability": str(values.get("answerability", "unknown")),
    }


def _canonical_valid_citations(answer, question, chunk_store) -> tuple[int, int, bool]:
    citations = getattr(answer, "citations", [])
    if not isinstance(citations, list):
        citations = []
    cited_ids = [
        citation.chunk_id
        for citation in citations
        if isinstance(getattr(citation, "chunk_id", None), str)
    ]
    unique_ids = list(dict.fromkeys(cited_ids))
    try:
        stored = chunk_store.get_chunks_by_ids(unique_ids)
    except Exception:
        stored = []
    canonical = {
        chunk.chunk_id: chunk
        for chunk in stored
        if isinstance(getattr(chunk, "chunk_id", None), str)
    }
    relevant = set(question.relevant_chunk_ids)
    valid_ids = {chunk_id for chunk_id in unique_ids if chunk_id in canonical and chunk_id in relevant}
    sufficient = bool(getattr(answer, "evidence_sufficient", False))
    return len(valid_ids), len(cited_ids), sufficient


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Evaluation Summary",
        "",
        f"- Timestamp (UTC): {payload['timestamp_utc']}",
        f"- Dataset fingerprint: `{payload['dataset_fingerprint']}`",
        f"- Acceptance: `{payload['acceptance']['code']}`",
        f"- Dataset questions: {payload['dataset_summary']['question_count']}",
        "- Categories: " + ", ".join(
            f"{name}={count}"
            for name, count in payload["dataset_summary"]["category_counts"].items()
        ),
        "",
        "| Mode | Evidence-group Recall@5 | MRR | Paper Recall@5 |",
        "|---|---:|---:|---:|",
    ]
    for mode in ("keyword", "vector", "hybrid", "two_stage"):
        if mode not in payload["metrics"]:
            continue
        metrics = payload["metrics"][mode]
        lines.append(
            f"| {mode} | {metrics['evidence_group_recall_at_5']:.6f} | {metrics['mrr']:.6f} | "
            f"{metrics['paper_recall_at_5']:.6f} |"
        )
    overall = payload["metrics"]["overall"]
    lines.extend(
        [
            "",
            f"- Citation precision: {overall['citation_precision']:.6f}",
            f"- Evidence coverage: {overall['evidence_coverage']:.6f}",
            f"- Unsupported claim rate: {overall['unsupported_claim_rate']:.6f}",
            "",
            "Question IDs: " + ", ".join(
                item["question_id"] for item in payload["questions"]
            ),
            "",
        ]
    )
    if (
        payload["dataset_summary"]["question_count"] < 20
        or payload["dataset_summary"]["evaluation_scope"] == "retrieval_only"
    ):
        lines.extend(
            [
                "",
                "注意：当前结果仅代表当前本地语料版本，不能作为通用线上准确率。",
            ]
        )
    return "\n".join(lines)


def _atomic_write(path: Path, contents: str) -> None:
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise EvaluationError("evaluation_report_write_failed") from None


def _report_paths(reports_dir: Path, stamp: str) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(100):
        suffix = secrets.token_hex(4)
        stem = f"evaluation-{stamp}-{suffix}"
        json_path = reports_dir / f"{stem}.json"
        markdown_path = reports_dir / f"{stem}.md"
        if not json_path.exists() and not markdown_path.exists():
            return json_path, markdown_path
    raise EvaluationError("evaluation_report_name_exhausted")


def _validated_config(config: dict[str, Any] | None) -> dict[str, Any]:
    default = {
        "retrieval_k": 5,
        "hybrid": {"keyword_weight": 2.0, "vector_weight": 0.5, "rrf_k": 60},
    }
    if config is None:
        return default
    if not isinstance(config, dict) or set(config) not in (
        {"retrieval_k", "hybrid"},
        {"retrieval_k", "hybrid", "two_stage"},
    ):
        raise EvaluationError("evaluation_config_invalid")
    hybrid = config.get("hybrid")
    if not isinstance(hybrid, dict) or set(hybrid) != {
        "keyword_weight",
        "vector_weight",
        "rrf_k",
    }:
        raise EvaluationError("evaluation_config_invalid")
    retrieval_k = config.get("retrieval_k")
    rrf_k = hybrid.get("rrf_k")
    weights = (hybrid.get("keyword_weight"), hybrid.get("vector_weight"))
    if (
        isinstance(retrieval_k, bool)
        or not isinstance(retrieval_k, int)
        or retrieval_k != 5
        or isinstance(rrf_k, bool)
        or not isinstance(rrf_k, int)
        or rrf_k < 0
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
            for value in weights
        )
    ):
        raise EvaluationError("evaluation_config_invalid")
    two_stage = config.get(
        "two_stage",
        {
            "paper_candidate_k": 50,
            "paper_k": 12,
            "chunk_candidate_k": 40,
            "max_chunks_per_paper": 2,
        },
    )
    if not isinstance(two_stage, dict) or set(two_stage) != {
        "paper_candidate_k", "paper_k", "chunk_candidate_k", "max_chunks_per_paper"
    }:
        raise EvaluationError("evaluation_config_invalid")
    limits = {
        "paper_candidate_k": (5, 200),
        "paper_k": (2, 50),
        "chunk_candidate_k": (5, 200),
        "max_chunks_per_paper": (1, 5),
    }
    for name, (lower, upper) in limits.items():
        value = two_stage.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
            raise EvaluationError("evaluation_config_invalid")
    validated = {
        "retrieval_k": retrieval_k,
        "hybrid": {
            "keyword_weight": float(hybrid["keyword_weight"]),
            "vector_weight": float(hybrid["vector_weight"]),
            "rrf_k": rrf_k,
        },
    }
    if "two_stage" in config:
        validated["two_stage"] = {name: int(two_stage[name]) for name in limits}
    return validated


def _two_stage_acceptance(metrics_by_category: dict[str, Any]) -> dict[str, Any]:
    hybrid = metrics_by_category.get("cross_paper", {}).get("hybrid", {})
    two_stage = metrics_by_category.get("cross_paper", {}).get("two_stage", {})
    natural_hybrid = metrics_by_category.get("natural_language", {}).get("hybrid", {})
    natural_two_stage = metrics_by_category.get("natural_language", {}).get("two_stage", {})
    checks = {
        "cross_paper_paper_recall_not_lower": two_stage.get("paper_recall_at_5", 0.0)
        >= hybrid.get("paper_recall_at_5", 0.0),
        "cross_paper_evidence_group_recall_not_lower": two_stage.get(
            "evidence_group_recall_at_5", 0.0
        )
        >= hybrid.get("evidence_group_recall_at_5", 0.0),
        "cross_paper_strict_improvement": (
            two_stage.get("paper_recall_at_5", 0.0)
            > hybrid.get("paper_recall_at_5", 0.0)
            or two_stage.get("evidence_group_recall_at_5", 0.0)
            > hybrid.get("evidence_group_recall_at_5", 0.0)
        ),
        "natural_language_evidence_group_recall_not_lower": natural_two_stage.get(
            "evidence_group_recall_at_5", 0.0
        )
        >= natural_hybrid.get("evidence_group_recall_at_5", 0.0)
        if natural_two_stage
        else False,
    }
    accepted = all(checks.values())
    return {
        "accepted": accepted,
        "code": "two_stage_promotion_gate_passed" if accepted else "two_stage_promotion_gate_failed",
        "checks": checks,
    }


def run_evaluation(
    dataset_path: str | Path,
    *,
    keyword_retriever,
    vector_retriever,
    hybrid_retriever=None,
    hybrid_factory=None,
    two_stage_retriever=None,
    two_stage_factory=None,
    qa_factory,
    chunk_store,
    reports_dir: str | Path = Path("data/reports/evaluation"),
    config: dict[str, Any] | None = None,
    now: Callable[[], datetime] | None = None,
    include_answer_metrics: bool = True,
) -> EvaluationResult:
    """Evaluate identical questions in the configured retrieval modes and emit ID-only reports.

    Citation precision counts relevant cited IDs that also exist in the trusted
    chunk store. Evidence coverage is the share of question/mode pairs with at
    least one such citation. Unsupported-claim rate is unsupported confident
    answers divided by all confident answers; insufficient fallbacks are excluded.
    """
    effective_config = _validated_config(config)
    questions = load_questions(dataset_path)
    category_counts: dict[str, int] = {}
    for question in questions:
        category_counts[question.category] = category_counts.get(question.category, 0) + 1
    if hybrid_factory is not None:
        hybrid_retriever = hybrid_factory(
            keyword_retriever,
            vector_retriever,
            **effective_config["hybrid"],
        )
    if hybrid_retriever is None:
        raise EvaluationError("evaluation_hybrid_required")
    modes = {
        "keyword": keyword_retriever,
        "vector": vector_retriever,
        "hybrid": hybrid_retriever,
    }
    if two_stage_factory is not None:
        two_stage_retriever = two_stage_factory(
            keyword_retriever,
            vector_retriever,
            hybrid_retriever,
            **effective_config.get("two_stage", {}),
        )
    if two_stage_retriever is not None:
        modes["two_stage"] = two_stage_retriever
    mode_metrics: dict[str, dict[str, float]] = {}
    category_values: dict[str, dict[str, dict[str, list[float]]]] = {
        question.category: {
            mode: {
                "evidence_group_recall_at_5": [],
                "mrr": [],
                "paper_recall_at_5": [],
            }
                for mode in modes
        }
        for question in questions
    }
    rows = {item.question_id: {"question_id": item.question_id, "modes": {}} for item in questions}
    valid_citations = total_citations = covered_pairs = 0
    confident_answers = unsupported_answers = 0
    try:
        for mode, retriever in modes.items():
            group_recalls: list[float] = []
            reciprocal_ranks: list[float] = []
            paper_recalls: list[float] = []
            qa = qa_factory(retriever) if include_answer_metrics else None
            for question in questions:
                retrieval_k = effective_config["retrieval_k"]
                ranked = retriever.search(question.question, k=retrieval_k)
                ranked_chunk_ids = _safe_chunk_ids(ranked)
                ranked_paper_ids = [item.paper_id for item in ranked]
                group_recall = evidence_group_recall_at_k(
                    ranked_chunk_ids,
                    [group.chunk_ids for group in question.evidence_groups],
                    retrieval_k,
                )
                rr = reciprocal_rank(ranked_chunk_ids, question.relevant_chunk_ids)
                paper_recall = recall_at_k(
                    ranked_paper_ids, question.relevant_paper_ids, retrieval_k
                )
                group_recalls.append(group_recall)
                reciprocal_ranks.append(rr)
                paper_recalls.append(paper_recall)
                category_metrics = category_values[question.category][mode]
                category_metrics["evidence_group_recall_at_5"].append(group_recall)
                category_metrics["mrr"].append(rr)
                category_metrics["paper_recall_at_5"].append(paper_recall)
                if include_answer_metrics:
                    answer = qa.answer(question.question)
                    valid, total, sufficient = _canonical_valid_citations(
                        answer, question, chunk_store
                    )
                    valid_citations += valid
                    total_citations += total
                    covered_pairs += int(valid > 0)
                    if sufficient:
                        confident_answers += 1
                        unsupported_answers += int(valid == 0)
                else:
                    valid = total = 0
                    sufficient = False
                mode_row = {
                    "evidence_group_recall_at_5": group_recall,
                    "mrr": rr,
                    "paper_recall_at_5": paper_recall,
                    "valid_citation_count": valid,
                    "citation_count": total,
                    "evidence_sufficient": sufficient,
                }
                trace = _safe_retrieval_trace(retriever)
                if trace is not None:
                    mode_row["retrieval_trace"] = trace
                answer_trace = _safe_qa_trace(qa) if include_answer_metrics else None
                if answer_trace is not None:
                    mode_row["answer_trace"] = answer_trace
                rows[question.question_id]["modes"][mode] = mode_row
            mode_metrics[mode] = {
                "evidence_group_recall_at_5": deterministic_mean(group_recalls),
                "mrr": deterministic_mean(reciprocal_ranks),
                "paper_recall_at_5": deterministic_mean(paper_recalls),
            }
    except EvaluationError:
        raise
    except Exception:
        raise EvaluationError("evaluation_service_failed") from None

    total_pairs = len(questions) * len(modes)
    metrics: dict[str, Any] = {
        **mode_metrics,
        "overall": {
            "citation_precision": citation_precision(valid_citations, total_citations),
            "evidence_coverage": evidence_coverage(covered_pairs, total_pairs),
            "unsupported_claim_rate": unsupported_claim_rate(
                unsupported_answers, confident_answers
            ),
        },
    }
    metrics_by_category = {
        category: {
            mode: {
                metric: deterministic_mean(values)
                for metric, values in metric_values.items()
            }
            for mode, metric_values in mode_values.items()
        }
        for category, mode_values in category_values.items()
    }
    regression = (
        metrics["hybrid"]["evidence_group_recall_at_5"]
        < metrics["keyword"]["evidence_group_recall_at_5"]
        and metrics["hybrid"]["evidence_group_recall_at_5"]
        < metrics["vector"]["evidence_group_recall_at_5"]
    )
    acceptance = (
        _two_stage_acceptance(metrics_by_category)
        if "two_stage" in modes
        else {
            "accepted": not regression,
            "code": "hybrid_recall_regression" if regression else "accepted",
        }
    )
    timestamp, stamp = _timestamp((now or (lambda: datetime.now(timezone.utc)))())
    payload = {
        "timestamp_utc": timestamp,
        "dataset_fingerprint": _fingerprint(questions),
        "config": effective_config,
        "metrics": metrics,
        "metrics_by_category": metrics_by_category,
        "dataset_summary": {
            "question_count": len(questions),
            "category_counts": dict(sorted(category_counts.items())),
            "evaluation_scope": (
                "answer_and_retrieval" if include_answer_metrics else "retrieval_only"
            ),
        },
        "questions": [rows[item.question_id] for item in questions],
        "acceptance": acceptance,
    }
    if not include_answer_metrics:
        payload["evaluation_scope"] = "retrieval_only"
    json_path, markdown_path = _report_paths(Path(reports_dir), stamp)
    _atomic_write(json_path, json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    try:
        _atomic_write(markdown_path, _markdown(payload))
    except EvaluationError:
        json_path.unlink(missing_ok=True)
        raise
    return EvaluationResult(json_path, markdown_path, metrics, acceptance)
