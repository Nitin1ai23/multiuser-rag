"""Mirror conversations from PostgreSQL to per-user object storage.

PostgreSQL stays the live store — it is what every read, the sidebar list, and
follow-up history hit. This module keeps a durable JSON copy of each
conversation in MinIO under its owner's prefix, rewritten whenever the
conversation changes (a message is added, it is renamed) and deleted with it.

The snapshot is *built* on the calling thread and only *uploaded* on a
background one. That split matters: the database connection is shared
process-wide, so reading it from the mirror threads would put concurrent
readers on one connection, and a network round-trip on the request thread would
slow every chat message. Upload failures are logged and dropped — losing the
archive copy must never fail a chat the user already sees on screen.

Uploads run on a *single* worker, which is what keeps the archive correct. Each
write replaces a conversation's whole object, so two concurrent uploads of the
same chat can finish out of order and leave an older snapshot as the final
state — a stale chat in storage, with nothing to correct it until the next
message. One worker runs tasks in submission order, so the newest snapshot is
always the last one written. Volume here is a few small JSON PUTs per message.
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from ..auth.db import get_db
from ..storage import chat_docs_prefix, chat_key, get_storage

logger = logging.getLogger(__name__)

_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def _submit(fn, *args) -> None:
    """Queue ``fn`` on the mirror worker, creating it on first use.

    Lazy so a deployment with storage disabled never starts the thread. Exactly
    one worker: see the module docstring — it is what orders the uploads.
    """
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="chat-mirror"
            )
        _executor.submit(_guarded, fn, *args)


def _guarded(fn, *args) -> None:
    try:
        fn(*args)
    except Exception as exc:  # noqa: BLE001 - a mirror failure must not surface
        logger.warning("Chat mirror failed: %s", exc)


def build_snapshot(user_id: str, conversation_id: str) -> dict | None:
    """Read one conversation out of PostgreSQL as a JSON-ready dict.

    Returns None if the conversation does not exist or is not this user's — the
    ownership check lives in the WHERE clause, like the rest of the chat layer.
    """
    db = get_db()
    conv = db.execute(
        "SELECT id, title, created_at, updated_at FROM conversations "
        "WHERE id = ? AND user_id = ?",
        (conversation_id, user_id),
    ).fetchone()
    if conv is None:
        return None
    rows = db.execute(
        "SELECT role, content, created_at FROM messages "
        "WHERE user_id = ? AND conversation_id = ? ORDER BY id ASC",
        (user_id, conversation_id),
    ).fetchall()
    return {
        "user_id": user_id,
        "conversation_id": conv["id"],
        "title": conv["title"],
        "created_at": conv["created_at"],
        "updated_at": conv["updated_at"],
        "messages": [
            {"role": r["role"], "content": r["content"], "created_at": r["created_at"]}
            for r in rows
        ],
    }


def mirror_conversation(user_id: str, conversation_id: str | None) -> None:
    """Queue an upload of this conversation's current state.

    A None ``conversation_id`` is the desktop app's legacy flat chat, which has
    no conversation row to snapshot, so it is skipped.
    """
    if not conversation_id or get_storage() is None:
        return
    snapshot = build_snapshot(user_id, conversation_id)
    if snapshot is None:
        return
    payload = json.dumps(snapshot, ensure_ascii=False, indent=2).encode("utf-8")
    _submit(_upload, chat_key(user_id, conversation_id), payload)


def _upload(key: str, payload: bytes) -> None:
    storage = get_storage()
    if storage is not None:
        storage.put_bytes(key, payload, content_type="application/json")


def remove_conversation(user_id: str, conversation_id: str) -> None:
    """Queue deletion of a conversation's snapshot and its document originals."""
    if not conversation_id or get_storage() is None:
        return
    _submit(_delete, user_id, conversation_id)


def _delete(user_id: str, conversation_id: str) -> None:
    storage = get_storage()
    if storage is None:
        return
    storage.delete(chat_key(user_id, conversation_id))
    storage.delete_prefix(chat_docs_prefix(user_id, conversation_id))


def flush() -> None:
    """Block until queued mirrors finish. For tests and orderly shutdown."""
    global _executor
    with _executor_lock:
        executor, _executor = _executor, None
    if executor is not None:
        executor.shutdown(wait=True)
