"""Per-user chat history backed by PostgreSQL.

Every method is scoped to a ``user_id``; there is no cross-user read path, so a
user only ever sees their own messages.

A history is optionally scoped to a single ``conversation_id`` (the web app's
selectable chats). When no conversation is given the history operates on the
user's *legacy* flat chat — the rows whose ``conversation_id`` is NULL — which
is what the PyQt5 desktop app uses.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..auth.db import get_db
from .mirror import mirror_conversation


@dataclass
class Message:
    role: str        # 'user' or 'assistant'
    content: str
    created_at: str


class ChatHistory:
    def __init__(self, user_id: str, conversation_id: str | None = None) -> None:
        if not user_id:
            raise ValueError("ChatHistory requires a user_id.")
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.db = get_db()

    def add(self, role: str, content: str) -> None:
        self.db.execute(
            "INSERT INTO messages (user_id, conversation_id, role, content) "
            "VALUES (?, ?, ?, ?)",
            (self.user_id, self.conversation_id, role, content),
        )
        if self.conversation_id is not None:
            # Keep the conversation list ordered by recent activity.
            self.db.execute(
                "UPDATE conversations "
                "SET updated_at = to_char((now() AT TIME ZONE 'UTC'), "
                "'YYYY-MM-DD HH24:MI:SS') WHERE id = ? AND user_id = ?",
                (self.conversation_id, self.user_id),
            )
        self.db.commit()
        # Archive the new state to object storage (no-op unless enabled), after
        # the commit so the snapshot can never contain an unwritten message.
        mirror_conversation(self.user_id, self.conversation_id)

    def all(self) -> list[Message]:
        if self.conversation_id is None:
            rows = self.db.execute(
                "SELECT role, content, created_at FROM messages "
                "WHERE user_id = ? AND conversation_id IS NULL ORDER BY id ASC",
                (self.user_id,),
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT role, content, created_at FROM messages "
                "WHERE user_id = ? AND conversation_id = ? ORDER BY id ASC",
                (self.user_id, self.conversation_id),
            ).fetchall()
        return [Message(r["role"], r["content"], r["created_at"]) for r in rows]

    def clear(self) -> None:
        if self.conversation_id is None:
            self.db.execute(
                "DELETE FROM messages WHERE user_id = ? AND conversation_id IS NULL",
                (self.user_id,),
            )
        else:
            self.db.execute(
                "DELETE FROM messages WHERE user_id = ? AND conversation_id = ?",
                (self.user_id, self.conversation_id),
            )
        self.db.commit()
        mirror_conversation(self.user_id, self.conversation_id)
