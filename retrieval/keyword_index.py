"""SQLite FTS5 keyword retrieval over stored evidence chunks."""

from __future__ import annotations

import sqlite3
import unicodedata
from dataclasses import dataclass

from domain.models import EvidenceChunk
from storage.database import ResearchDatabase

from retrieval.lexical_plan import lexical_focus
from retrieval.query_expansion import plan_query


class KeywordSearchError(RuntimeError):
    """Stable boundary for database/index failures unrelated to user syntax."""


@dataclass(frozen=True)
class KeywordTrace:
    status: str = "not_started"
    focused: bool = False
    planned_branches: int = 0
    sql_calls: int = 0
    fallback_calls: int = 0
    candidate_window: int = 0
    selected_count: int = 0
    ranking_mode: str = "not_started"


def _safe_fts_query(query: str, *, operator: str = "AND") -> str:
    terms: list[str] = []
    current: list[str] = []
    for character in unicodedata.normalize("NFKC", query):
        category = unicodedata.category(character)
        if category[0] in {"L", "N"} or character == "_":
            current.append(character)
        elif current:
            terms.append("".join(current))
            current = []
    if current:
        terms.append("".join(current))
    terms = [
        term
        for term in terms
        if len(term) > 1 or not term.isascii() or term.isdigit()
    ]
    if operator not in {"AND", "OR"}:
        raise ValueError("keyword_operator_invalid")
    return f" {operator} ".join(f'"{term}"' for term in terms)


class KeywordIndex:
    """Search the transactionally synchronized FTS projection."""

    def __init__(self, database: ResearchDatabase) -> None:
        self.database = database
        self.last_trace = KeywordTrace()

    def search(
        self,
        query: str,
        k: int,
        paper_ids: list[str] | None = None,
    ) -> list[EvidenceChunk]:
        self.last_trace = KeywordTrace(status="empty")
        if k <= 0 or paper_ids == []:
            return []
        # BM25 keeps the original wording plus constrained keyword variants;
        # dense-only semantic rewrites are left to the vector branch.
        queries = plan_query(query).keyword_queries
        if not queries:
            return []
        sql = """
            SELECT chunks.*, bm25(chunk_fts) AS rank
            FROM chunk_fts
            JOIN chunks ON chunks.chunk_id = chunk_fts.chunk_id
            WHERE chunk_fts MATCH ?
        """
        rows_by_id = {}
        scores = {}
        seen_expressions = set()
        focus = lexical_focus(query)
        groups = ["(" + " OR ".join(_safe_fts_query(term) for term in alternatives) + ")"
                  for alternatives in focus.concepts]
        entities = [_safe_fts_query(term) for term in focus.entities]
        focused = " AND ".join([*entities, *groups])
        focused_fallback = " AND ".join([*entities, "(" + " OR ".join(groups) + ")"])
        # Fusion needs candidates beyond the final display cutoff. Keep the
        # window bounded, consistent with the existing hybrid candidate stage.
        candidate_window = min(100, max(k, 20)) if groups else k
        sql_calls = fallback_calls = 0
        branches = []
        if groups:
            branches.append((focused, focused_fallback, 2.0))
        branches.extend((_safe_fts_query(expanded), _safe_fts_query(expanded, operator="OR"), 1.0)
                        for expanded in queries
                        if not (groups and expanded.endswith(" method dataset metric evidence")))
        try:
            with self.database._connect() as connection:
                for fts_query, fallback_query, weight in branches:
                    if not fts_query or fts_query in seen_expressions:
                        continue
                    seen_expressions.add(fts_query)
                    parameters: list[object] = [fts_query]
                    scoped_sql = sql
                    if paper_ids is not None:
                        placeholders = ", ".join("?" for _ in paper_ids)
                        scoped_sql += f" AND chunks.paper_id IN ({placeholders})"
                        parameters.extend(paper_ids)
                    scoped_sql += " ORDER BY rank ASC, chunks.chunk_id ASC LIMIT ?"
                    parameters.append(candidate_window)
                    sql_calls += 1
                    rows = connection.execute(scoped_sql, parameters).fetchall()
                    if not rows and fallback_query != fts_query and fallback_query not in seen_expressions:
                        seen_expressions.add(fallback_query)
                        fallback_parameters = [fallback_query, *parameters[1:]]
                        sql_calls += 1
                        fallback_calls += 1
                        rows = connection.execute(scoped_sql, fallback_parameters).fetchall()
                    # BM25 orders each query independently. Fuse ranks, not raw
                    # magnitudes from different term sets or AND/OR fallbacks.
                    for rank, row in enumerate(rows, 1):
                        key = row["chunk_id"]
                        rows_by_id.setdefault(key, row)
                        if groups:
                            scores[key] = scores.get(key, 0.0) + weight / (60 + rank)
                        else:
                            # Preserve the established non-focused path until a
                            # separate comparison justifies migrating its scores.
                            scores[key] = max(scores.get(key, 0.0), -float(row["rank"]))
        except sqlite3.Error:
            self.last_trace = KeywordTrace(status="failed", focused=bool(groups), planned_branches=len(branches),
                                           sql_calls=sql_calls, fallback_calls=fallback_calls, candidate_window=candidate_window)
            raise KeywordSearchError("keyword_search_failed") from None
        rows = sorted(rows_by_id.values(), key=lambda row: (-scores[row["chunk_id"]], row["chunk_id"]))[:k]
        self.last_trace = KeywordTrace(status="finished", focused=bool(groups), planned_branches=len(branches),
                                       sql_calls=sql_calls, fallback_calls=fallback_calls, candidate_window=candidate_window,
                                       selected_count=len(rows), ranking_mode="focused_rrf" if groups else "legacy_bm25")
        return [
            EvidenceChunk(
                chunk_id=row["chunk_id"],
                paper_id=row["paper_id"],
                title=row["title"],
                page_number=row["page_number"],
                section=row["section"],
                text=row["text"],
                score=scores[row["chunk_id"]],
            )
            for row in rows
        ]
