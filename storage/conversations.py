"""Local chat archive, separate from the bounded context sent to the model."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from domain.models import ResearchConversationMessage, ResearchConversationState


class ConversationStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS desktop_conversations (
                id TEXT PRIMARY KEY, title TEXT NOT NULL, state TEXT NOT NULL,
                transcript TEXT NOT NULL, updated_at TEXT NOT NULL)""")

    def list(self, query: str = "") -> list[dict]:
        with closing(sqlite3.connect(self.path)) as db:
            rows = db.execute(
                "SELECT id, title, updated_at FROM desktop_conversations ORDER BY updated_at DESC"
            ).fetchall()
        return [{"id": row[0], "title": row[1], "updated_at": row[2]}
                for row in rows if query.casefold() in row[1].casefold()]

    def load(self, conversation_id: str) -> tuple[ResearchConversationState, list[ResearchConversationMessage]]:
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute("SELECT state, transcript FROM desktop_conversations WHERE id = ?",
                             (conversation_id,)).fetchone()
        if row is None:
            return ResearchConversationState(), []
        return (ResearchConversationState.model_validate_json(row[0]),
                [ResearchConversationMessage.model_validate(item) for item in json.loads(row[1])])

    def save(self, conversation_id: str, state: ResearchConversationState,
             transcript: list[ResearchConversationMessage]) -> None:
        title = next((m.content.strip()[:48] for m in transcript if m.role == "user"), "新对话")
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("""INSERT INTO desktop_conversations VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET title=excluded.title, state=excluded.state,
                transcript=excluded.transcript, updated_at=excluded.updated_at""",
                (conversation_id, title, state.model_dump_json(),
                 json.dumps([m.model_dump(mode="json") for m in transcript], ensure_ascii=False),
                 datetime.now(UTC).isoformat()))

    def forget_context(self, conversation_id: str) -> None:
        # History stays visible, but no previous messages, summary or scope are
        # sent to subsequent model calls. Project memory has a separate control.
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("UPDATE desktop_conversations SET state = ? WHERE id = ?",
                       (ResearchConversationState().model_dump_json(), conversation_id))

    def delete(self, conversation_id: str) -> None:
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("DELETE FROM desktop_conversations WHERE id = ?", (conversation_id,))
