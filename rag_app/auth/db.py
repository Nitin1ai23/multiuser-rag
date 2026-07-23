"""PostgreSQL database for users and chat history.

The rest of the app was written against the small ``sqlite3.Connection`` surface
it needs — ``conn.execute(sql, params)`` returning a cursor you can immediately
``.fetchone()``/``.fetchall()``, plus ``.commit()`` and rows indexable by column
name. :class:`Database` reproduces exactly that surface on top of psycopg 3 so
the call sites (auth, chat history, conversations, mirror, token revocation) stay
unchanged apart from Postgres SQL dialect:

* ``?`` placeholders are rewritten to psycopg's ``%s`` on the way through
  (no SQL in this app contains a literal ``?`` or ``%``, so the swap is safe);
* rows come back as dicts (``row["column"]``) via psycopg's ``dict_row``;
* each thread gets its own connection (``threading.local``), because FastAPI
  runs sync endpoints in a threadpool and the chat mirror runs on a background
  thread — a single shared psycopg connection is not safe for concurrent use,
  whereas one-per-thread keeps every statement on a connection only that thread
  touches.

Connections run in **autocommit** mode. The call sites commit only after a
write and never after a read (matching SQLite, where a bare ``SELECT`` starts no
lasting transaction); under psycopg's default those reads would sit
"idle in transaction" holding a snapshot and locks. Autocommit makes each
statement durable on its own — ``.commit()`` becomes a harmless no-op — which is
the behaviour this code was written against. The few multi-statement writes
(deleting an account or a conversation) are covered by ``ON DELETE CASCADE``, so
they stay consistent without an explicit transaction.
"""

from __future__ import annotations

import threading
from functools import lru_cache

import psycopg
from psycopg.rows import dict_row

from ..config import get_settings

# Raised by a duplicate-key insert (username/email uniqueness). Re-exported so
# callers depend on this module rather than importing psycopg directly.
IntegrityError = psycopg.errors.UniqueViolation

# ``id INTEGER PRIMARY KEY AUTOINCREMENT`` (SQLite) becomes an identity column;
# ``datetime('now')`` becomes a UTC wall-clock string so created_at/updated_at
# keep the exact 'YYYY-MM-DD HH24:MI:SS' text format the API and UI expect.
_NOW = "to_char((now() AT TIME ZONE 'UTC'), 'YYYY-MM-DD HH24:MI:SS')"

# Postgres cannot run multiple statements in one parameterised execute, so the
# schema is a list of individual statements applied in order. All are
# idempotent (IF NOT EXISTS), so connecting to an existing database is a no-op.
SCHEMA: tuple[str, ...] = (
    f"""
    CREATE TABLE IF NOT EXISTS users (
        id                   TEXT PRIMARY KEY,
        username             TEXT NOT NULL UNIQUE,
        email                TEXT NOT NULL UNIQUE,
        password_hash        TEXT NOT NULL,
        password_salt        TEXT NOT NULL,
        iterations           INTEGER NOT NULL,
        security_question    TEXT NOT NULL,
        security_answer_hash TEXT NOT NULL,
        security_answer_salt TEXT NOT NULL,
        created_at           TEXT NOT NULL DEFAULT {_NOW}
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS conversations (
        id         TEXT PRIMARY KEY,
        user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        title      TEXT NOT NULL DEFAULT 'New chat',
        created_at TEXT NOT NULL DEFAULT {_NOW},
        updated_at TEXT NOT NULL DEFAULT {_NOW}
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_conversations_user
        ON conversations(user_id, updated_at)
    """,
    f"""
    CREATE TABLE IF NOT EXISTS messages (
        id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
        role            TEXT NOT NULL,
        content         TEXT NOT NULL,
        created_at      TEXT NOT NULL DEFAULT {_NOW}
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, id)",
    """
    CREATE INDEX IF NOT EXISTS idx_messages_conversation
        ON messages(conversation_id, id)
    """,
    """
    CREATE TABLE IF NOT EXISTS revoked_tokens (
        jti TEXT PRIMARY KEY,
        exp BIGINT NOT NULL
    )
    """,
)


class Database:
    """A thin, thread-safe stand-in for the ``sqlite3.Connection`` the app used.

    One psycopg connection is opened lazily per thread. The public surface is
    intentionally small: :meth:`execute`, :meth:`commit`, :meth:`rollback`,
    :meth:`close` — everything the call sites use.
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._local = threading.local()

    def _conn(self) -> psycopg.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None or conn.closed:
            conn = psycopg.connect(self._dsn, row_factory=dict_row, autocommit=True)
            self._local.conn = conn
        return conn

    def execute(self, sql: str, params: tuple = ()) -> psycopg.Cursor:
        """Run one statement and return its cursor (so ``.fetchone()`` chains)."""
        return self._conn().execute(sql.replace("?", "%s"), params)

    def commit(self) -> None:
        self._conn().commit()

    def rollback(self) -> None:
        """Clear an aborted transaction (Postgres blocks further work until then)."""
        self._conn().rollback()

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


def _connect() -> Database:
    settings = get_settings()
    db = Database(settings.database_url)
    for statement in SCHEMA:
        db.execute(statement)
    db.commit()
    return db


@lru_cache
def get_db() -> Database:
    """Return the process-wide database handle (one connection per thread)."""
    return _connect()
