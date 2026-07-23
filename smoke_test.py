"""Offline smoke test: auth + per-user isolation (no API keys needed).

Uses a throwaway Postgres database and data dir plus fake vectors, so we never
call NIM/Groq and never touch the real ``ragdb``. Needs a running Postgres
(``docker compose up -d postgres``); the throwaway database is created here and
dropped on exit.
"""

import os
import shutil
import tempfile

import psycopg

tmp = tempfile.mkdtemp(prefix="ragtest_")
os.environ["QDRANT_PATH"] = os.path.join(tmp, "qdrant")
os.environ["EMBEDDING_DIM"] = "8"  # tiny vectors for the test
# Keys intentionally absent — we never call NIM/Groq here.

# A uniquely named throwaway database so a smoke run never collides with real
# data or a parallel run. Built from the configured DATABASE_URL (host, port,
# credentials) with the database name swapped out, then dropped in `finally`.
_base = psycopg.conninfo.conninfo_to_dict(
    os.environ.get(
        "DATABASE_URL", "postgresql://raguser:ragpass@localhost:5434/ragdb"
    )
)
_smoke_db = f"ragsmoke_{os.getpid()}"
_admin = psycopg.conninfo.make_conninfo(**{**_base, "dbname": "postgres"})
with psycopg.connect(_admin, autocommit=True) as _c:
    _c.execute(f'DROP DATABASE IF EXISTS "{_smoke_db}"')
    _c.execute(f'CREATE DATABASE "{_smoke_db}"')
os.environ["DATABASE_URL"] = psycopg.conninfo.make_conninfo(
    **{**_base, "dbname": _smoke_db}
)

from rag_app.config import get_settings
from rag_app.auth.service import AuthService, AuthError
from rag_app.chat.history import ChatHistory
from rag_app.vectorstore import get_store

# Two of the preset questions signup accepts (see AuthService.SECURITY_QUESTIONS).
Q_PET = "What was the name of your first pet?"
Q_CITY = "In what city were you born?"

DIM = get_settings().embedding_dim


def vec(seed: float):
    return [seed] * DIM


def main() -> None:
    auth = AuthService()

    # --- signup + login ---------------------------------------------------
    alice = auth.signup("alice", "alice@x.com", "password123", Q_PET, "Rex")
    bob = auth.signup("bob", "bob@x.com", "password456", Q_CITY, "Paris")
    assert alice.id != bob.id

    assert auth.login("alice", "password123").id == alice.id
    assert auth.login("alice@x.com", "password123").id == alice.id  # email login
    try:
        auth.login("alice", "wrong")
        raise SystemExit("FAIL: wrong password accepted")
    except AuthError:
        pass

    # duplicate username rejected
    try:
        auth.signup("alice", "other@x.com", "password789", Q_PET, "a")
        raise SystemExit("FAIL: duplicate username accepted")
    except AuthError:
        pass

    # --- forgot password via security question ----------------------------
    assert auth.get_security_question("bob") == Q_CITY
    try:
        auth.reset_password("bob", "WrongCity", "newpass12")
        raise SystemExit("FAIL: wrong security answer accepted")
    except AuthError:
        pass
    auth.reset_password("bob", "paris", "newpass12")  # case-insensitive answer
    assert auth.login("bob", "newpass12").id == bob.id

    # --- vector store isolation ------------------------------------------
    store = get_store()
    store.upsert(["alice secret doc"], [vec(0.1)], user_id=alice.id, source="a.txt")
    store.upsert(["bob secret doc"], [vec(0.9)], user_id=bob.id, source="b.txt")

    a_hits = store.search(vec(0.1), user_id=alice.id, top_k=10)
    b_hits = store.search(vec(0.9), user_id=bob.id, top_k=10)
    assert {h.text for h in a_hits} == {"alice secret doc"}, a_hits
    assert {h.text for h in b_hits} == {"bob secret doc"}, b_hits
    # Alice must NEVER see bob's chunk even querying with his vector:
    a_all = store.search(vec(0.9), user_id=alice.id, top_k=10)
    assert all(h.text != "bob secret doc" for h in a_all), "ISOLATION BREACH"

    assert store.list_sources(alice.id) == [("a.txt", 1)]
    assert store.list_sources(bob.id) == [("b.txt", 1)]

    # --- chat history isolation ------------------------------------------
    ChatHistory(alice.id).add("user", "hi from alice")
    ChatHistory(bob.id).add("user", "hi from bob")
    a_msgs = [m.content for m in ChatHistory(alice.id).all()]
    b_msgs = [m.content for m in ChatHistory(bob.id).all()]
    assert a_msgs == ["hi from alice"], a_msgs
    assert b_msgs == ["hi from bob"], b_msgs
    ChatHistory(alice.id).clear()
    assert ChatHistory(alice.id).all() == []
    assert store.list_sources(alice.id) == [("a.txt", 1)]

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        # Close the app's connection(s) before dropping the throwaway database,
        # then remove it so a smoke run leaves nothing behind.
        try:
            from rag_app.auth.db import get_db

            get_db().close()
        except Exception:
            pass
        with psycopg.connect(_admin, autocommit=True) as _c:
            _c.execute(f'DROP DATABASE IF EXISTS "{_smoke_db}" WITH (FORCE)')
