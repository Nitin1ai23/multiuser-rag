"""Per-user object storage: key layout, isolation, and the chat mirror.

No MinIO runs here — the key helpers are pure, and the mirror is driven against
a fake store that records what it was asked to write.
"""

import json
import time

import pytest

from rag_app import storage
from rag_app.auth.service import SECURITY_QUESTIONS, AuthService
from rag_app.chat import mirror
from rag_app.chat.conversations import ConversationStore
from rag_app.chat.history import ChatHistory
from rag_app.storage import (
    StorageError,
    chat_docs_prefix,
    chat_key,
    document_key,
    user_prefix,
)


class FakeStore:
    """Records writes/deletes in a dict, standing in for ObjectStore."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def put_bytes(self, key, data, content_type=None):
        self.objects[key] = data

    def put_file(self, key, path):
        with open(path, "rb") as fh:
            self.objects[key] = fh.read()

    def get_bytes(self, key):
        return self.objects.get(key)

    def list_keys(self, prefix):
        return [k for k in self.objects if k.startswith(prefix)]

    def delete(self, key):
        self.objects.pop(key, None)

    def delete_prefix(self, prefix):
        keys = self.list_keys(prefix)
        for k in keys:
            del self.objects[k]
        return len(keys)


class _StubPipeline:
    """Stands in for RAGPipeline: constructing the real one needs API keys."""

    def __init__(self, user_id, conversation_id=None):
        pass

    def ingest_file(self, path):
        return 3


@pytest.fixture
def fake_store(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(mirror, "get_storage", lambda: store)
    yield store
    mirror.flush()


def _new_user(username="alice"):
    return AuthService().signup(
        username=username,
        email=f"{username}@example.com",
        password="password123",
        security_question=SECURITY_QUESTIONS[0],
        security_answer="Rex",
    )


# --- Key layout ---------------------------------------------------------
def test_keys_are_namespaced_under_the_owning_user():
    assert user_prefix("u1") == "users/u1/"
    assert chat_key("u1", "c1") == "users/u1/chats/c1.json"
    assert chat_docs_prefix("u1", "c1") == "users/u1/chats/c1/docs/"
    assert document_key("u1", "c1", "report.pdf") == "users/u1/chats/c1/docs/report.pdf"
    # Two users' keys never overlap, so one user's prefix cannot address another's.
    assert not chat_key("u2", "c1").startswith(user_prefix("u1"))


def test_document_key_keeps_archive_member_paths():
    key = document_key("u1", "c1", "archive.zip/inner/file.txt")
    assert key == "users/u1/chats/c1/docs/archive.zip/inner/file.txt"


@pytest.mark.parametrize("bad", ["", "../u2", "u1/../u2", "a/b"])
def test_key_helpers_reject_ids_that_escape_the_user_prefix(bad):
    with pytest.raises(StorageError):
        user_prefix(bad)


@pytest.mark.parametrize("bad", ["", "../../etc/passwd", "a/../../b"])
def test_document_key_rejects_traversal_in_the_source(bad):
    with pytest.raises(StorageError):
        document_key("u1", "c1", bad)


# --- Disabled by default ------------------------------------------------
def test_storage_is_disabled_without_configuration():
    assert storage.get_storage() is None


def test_mirror_is_a_no_op_when_storage_is_disabled():
    user = _new_user()
    conv = ConversationStore(user.id).create()
    ChatHistory(user.id, conv.id).add("user", "hello")  # must not raise


# --- Chat mirror --------------------------------------------------------
def test_adding_a_message_mirrors_the_conversation(fake_store):
    user = _new_user()
    conv = ConversationStore(user.id).create("Trip planning")
    history = ChatHistory(user.id, conv.id)
    history.add("user", "where to?")
    history.add("assistant", "somewhere warm")
    mirror.flush()

    snapshot = json.loads(fake_store.objects[chat_key(user.id, conv.id)])
    assert snapshot["user_id"] == user.id
    assert snapshot["title"] == "Trip planning"
    assert [(m["role"], m["content"]) for m in snapshot["messages"]] == [
        ("user", "where to?"),
        ("assistant", "somewhere warm"),
    ]


def test_renaming_a_conversation_updates_the_snapshot(fake_store):
    user = _new_user()
    store = ConversationStore(user.id)
    conv = store.create()
    store.rename(conv.id, "Renamed")
    mirror.flush()

    snapshot = json.loads(fake_store.objects[chat_key(user.id, conv.id)])
    assert snapshot["title"] == "Renamed"


def test_deleting_a_conversation_removes_its_snapshot_and_documents(fake_store):
    user = _new_user()
    store = ConversationStore(user.id)
    conv = store.create()
    ChatHistory(user.id, conv.id).add("user", "hi")
    fake_store.put_bytes(document_key(user.id, conv.id, "notes.txt"), b"stuff")
    mirror.flush()

    store.delete(conv.id)
    mirror.flush()
    assert fake_store.objects == {}


def test_a_slow_upload_cannot_leave_a_stale_snapshot_behind(monkeypatch):
    """Snapshots of one chat must land in the order they were taken.

    Each write replaces the whole object, so if a slow earlier upload overlaps
    a later one it can finish last and pin storage to a stale version of the
    chat. Delaying the first write makes that reordering happen if uploads are
    ever run concurrently.
    """
    class SlowFirstStore(FakeStore):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def put_bytes(self, key, data, content_type=None):
            self.calls += 1
            if self.calls == 1:
                time.sleep(0.2)
            super().put_bytes(key, data, content_type)

    store = SlowFirstStore()
    monkeypatch.setattr(mirror, "get_storage", lambda: store)
    try:
        user = _new_user()
        conversations = ConversationStore(user.id)
        conv = conversations.create()          # first, slow write
        conversations.rename(conv.id, "Final title")  # second, fast write
        mirror.flush()
    finally:
        mirror.flush()

    snapshot = json.loads(store.objects[chat_key(user.id, conv.id)])
    assert snapshot["title"] == "Final title"


def test_mirror_ignores_the_legacy_flat_desktop_chat(fake_store):
    user = _new_user()
    ChatHistory(user.id).add("user", "desktop message")  # no conversation_id
    mirror.flush()
    assert fake_store.objects == {}


def test_mirror_never_writes_another_users_conversation(fake_store):
    alice = _new_user("alice")
    bob = _new_user("bob")
    conv = ConversationStore(alice.id).create()
    ChatHistory(alice.id, conv.id).add("user", "alice's secret")
    mirror.flush()

    # Bob naming Alice's conversation id snapshots nothing: the ownership check
    # is in the query, so there is no row to write under his prefix.
    mirror.mirror_conversation(bob.id, conv.id)
    mirror.flush()
    assert list(fake_store.objects) == [chat_key(alice.id, conv.id)]


# --- Document originals -------------------------------------------------
def test_ingest_job_archives_the_original_under_the_users_prefix(monkeypatch, tmp_path):
    from rag_app.web import ingest_jobs

    fake = FakeStore()
    monkeypatch.setattr(ingest_jobs, "get_storage", lambda: fake)
    monkeypatch.setattr(ingest_jobs, "RAGPipeline", _StubPipeline)

    src = tmp_path / "report.pdf"
    src.write_bytes(b"%PDF-1.4 fake")
    job = ingest_jobs.create_job("u1", "report.pdf", conversation_id="c1")
    ingest_jobs.run_ingest(job.id, str(src))

    assert job.status == "done"
    assert fake.objects[document_key("u1", "c1", "report.pdf")] == b"%PDF-1.4 fake"


def test_ingestion_survives_a_storage_outage(monkeypatch, tmp_path):
    """A MinIO failure costs the archived original, not the ability to ask."""
    from rag_app.web import ingest_jobs

    class Broken(FakeStore):
        def put_file(self, key, path):
            raise StorageError("connection refused")

    monkeypatch.setattr(ingest_jobs, "get_storage", lambda: Broken())
    monkeypatch.setattr(ingest_jobs, "RAGPipeline", _StubPipeline)

    src = tmp_path / "notes.txt"
    src.write_text("hello")
    job = ingest_jobs.create_job("u1", "notes.txt", conversation_id="c1")
    ingest_jobs.run_ingest(job.id, str(src))

    assert job.status == "done"
    assert job.chunks_added == 3
    assert "could not be saved" in job.detail
