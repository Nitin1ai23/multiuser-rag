"""Per-user conversations (chat sessions).

A user owns many conversations; each conversation owns many messages
(``rag_app/chat/history.py``). Every method is scoped to a ``user_id`` and any
method that takes a ``conversation_id`` re-checks ownership in the WHERE clause,
so a user can only ever see or mutate their own conversations.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from ..auth.db import get_db
from .mirror import mirror_conversation, remove_conversation

DEFAULT_TITLE = "New chat"
_TITLE_MAX = 60


@dataclass
class Conversation:
    id: str
    title: str
    created_at: str
    updated_at: str


def make_title(text: str) -> str:
    """Derive a short conversation title from the first user message."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return DEFAULT_TITLE
    return cleaned[:_TITLE_MAX].rstrip() + ("…" if len(cleaned) > _TITLE_MAX else "")


class ConversationStore:
    def __init__(self, user_id: str) -> None:
        if not user_id:
            raise ValueError("ConversationStore requires a user_id.")
        self.user_id = user_id
        self.db = get_db()

    def _row_to_conv(self, row) -> Conversation:
        return Conversation(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def list(self) -> list[Conversation]:
        """Most-recently-active conversation first."""
        rows = self.db.execute(
            "SELECT id, title, created_at, updated_at FROM conversations "
            "WHERE user_id = ? ORDER BY updated_at DESC, id DESC",
            (self.user_id,),
        ).fetchall()
        return [self._row_to_conv(r) for r in rows]

    def get(self, conversation_id: str) -> Conversation | None:
        row = self.db.execute(
            "SELECT id, title, created_at, updated_at FROM conversations "
            "WHERE id = ? AND user_id = ?",
            (conversation_id, self.user_id),
        ).fetchone()
        return self._row_to_conv(row) if row else None

    def create(self, title: str | None = None) -> Conversation:
        cid = str(uuid.uuid4())
        self.db.execute(
            "INSERT INTO conversations (id, user_id, title) VALUES (?, ?, ?)",
            (cid, self.user_id, (title or DEFAULT_TITLE)),
        )
        self.db.commit()
        mirror_conversation(self.user_id, cid)
        return self.get(cid)  # type: ignore[return-value]

    def rename(self, conversation_id: str, title: str) -> None:
        self.db.execute(
            "UPDATE conversations SET title = ? WHERE id = ? AND user_id = ?",
            (title, conversation_id, self.user_id),
        )
        self.db.commit()
        mirror_conversation(self.user_id, conversation_id)

    def touch(self, conversation_id: str) -> None:
        """Bump ``updated_at`` so the conversation floats to the top of the list."""
        self.db.execute(
            "UPDATE conversations "
            "SET updated_at = to_char((now() AT TIME ZONE 'UTC'), "
            "'YYYY-MM-DD HH24:MI:SS') WHERE id = ? AND user_id = ?",
            (conversation_id, self.user_id),
        )
        self.db.commit()

    def delete(self, conversation_id: str) -> bool:
        """Delete a conversation and its messages. Returns True if it existed.

        Messages are removed explicitly rather than relying on ``ON DELETE
        CASCADE``: databases upgraded via ``ALTER TABLE`` carry a
        ``conversation_id`` column with no foreign-key constraint, so the
        cascade would not fire there.
        """
        self.db.execute(
            "DELETE FROM messages WHERE conversation_id = ? AND user_id = ?",
            (conversation_id, self.user_id),
        )
        cur = self.db.execute(
            "DELETE FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, self.user_id),
        )
        self.db.commit()
        if cur.rowcount == 0:
            return False
        # Drop the archived snapshot and the originals of documents uploaded
        # into this chat, so deleting a chat deletes it everywhere.
        remove_conversation(self.user_id, conversation_id)
        return True
