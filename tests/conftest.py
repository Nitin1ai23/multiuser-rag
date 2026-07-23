"""Test fixtures: point every cached singleton at a throwaway test environment.

``Settings``, the database handle, and the Qdrant store are all ``lru_cache``d,
so each test clears those caches after pointing the relevant env vars at a
per-test temp directory (Qdrant) and a dedicated Postgres **test database**.

The suite needs a running Postgres — ``docker compose up -d postgres`` locally,
or the ``postgres`` service in CI. Point it elsewhere with ``TEST_DATABASE_URL``.
The test database is created once if missing, then every table is truncated
before each test so cases never see each other's rows. No API keys are required
— the tests exercise auth, chunking, isolation, and rate limiting, none of which
call out to NIM/Groq.
"""

from __future__ import annotations

import os

import psycopg
import pytest


def _default_test_url() -> str:
    """Same server/credentials as the app's DATABASE_URL, but a `*_test` database.

    Deriving it from the configured URL keeps the tests pointed at whatever
    Postgres the app uses (port, host, credentials) without a second place to
    update. Kept apart from the real database so a run never truncates live data.
    """
    from rag_app.config import get_settings

    info = psycopg.conninfo.conninfo_to_dict(get_settings().database_url)
    info["dbname"] = f"{info.get('dbname', 'ragdb')}_test"
    return psycopg.conninfo.make_conninfo(**info)


# Override the whole DSN with TEST_DATABASE_URL (CI sets this explicitly).
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL") or _default_test_url()

_TABLES = "users, conversations, messages, revoked_tokens"


def _ensure_database(dsn: str) -> None:
    """Create the test database if it does not exist yet.

    ``CREATE DATABASE`` can't run inside a transaction, so this connects to the
    ``postgres`` maintenance database in autocommit mode.
    """
    info = psycopg.conninfo.conninfo_to_dict(dsn)
    dbname = info["dbname"]
    admin = psycopg.conninfo.make_conninfo(**{**info, "dbname": "postgres"})
    with psycopg.connect(admin, autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)
        ).fetchone()
        if not exists:
            # dbname is our own constant, not user input — safe to interpolate.
            conn.execute(f'CREATE DATABASE "{dbname}"')


@pytest.fixture(scope="session", autouse=True)
def _test_database() -> None:
    """Make sure the test database exists before any test runs."""
    _ensure_database(TEST_DATABASE_URL)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("QDRANT_URL", "")
    monkeypatch.setenv("EMBEDDING_DIM", "8")          # tiny vectors for tests
    monkeypatch.setenv("PBKDF2_ITERATIONS", "1000")   # keep hashing fast
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "20")
    monkeypatch.setenv("RERANK_ENABLED", "false")
    monkeypatch.setenv("NVIDIA_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("STORAGE_ENABLED", "false")    # no MinIO in tests

    from rag_app import storage, vectorstore
    from rag_app.auth import db as authdb
    from rag_app.chat import mirror
    from rag_app.config import get_settings

    get_settings.cache_clear()
    authdb.get_db.cache_clear()
    vectorstore.get_store.cache_clear()
    storage.get_storage.cache_clear()

    # Fresh slate: creating the handle applies the schema, then truncate every
    # table (RESTART IDENTITY resets the messages id sequence between tests).
    db = authdb.get_db()
    db.execute(f"TRUNCATE {_TABLES} RESTART IDENTITY CASCADE")

    yield

    # Let queued chat mirrors finish before the connection they read closes.
    mirror.flush()
    try:
        authdb.get_db().close()
    except Exception:
        pass
    authdb.get_db.cache_clear()
    vectorstore.get_store.cache_clear()
    storage.get_storage.cache_clear()
    get_settings.cache_clear()
