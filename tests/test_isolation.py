"""The core multi-tenant guarantee: a user can never retrieve another's chunks,
even when searching with the other user's own vector."""

from rag_app.vectorstore import get_store

# 8-dim vectors (EMBEDDING_DIM=8 in the test env), one basis direction each.
VEC_A = [1.0, 0, 0, 0, 0, 0, 0, 0]
VEC_B = [0, 1.0, 0, 0, 0, 0, 0, 0]


def _seed(store):
    store.upsert(["alice's private note"], [VEC_A], user_id="alice", source="a.txt")
    store.upsert(["bob's private note"], [VEC_B], user_id="bob", source="b.txt")


def test_search_is_scoped_even_with_other_users_vector():
    store = get_store()
    _seed(store)
    # Alice queries with Bob's exact vector — she must still only see her own data.
    hits = store.search(VEC_B, user_id="alice", top_k=10)
    assert {h.source for h in hits} <= {"a.txt"}
    assert all("bob" not in h.text for h in hits)


def test_list_sources_is_per_user():
    store = get_store()
    _seed(store)
    assert store.list_sources("alice") == [("a.txt", 1)]
    assert store.list_sources("bob") == [("b.txt", 1)]


def test_delete_only_affects_owner():
    store = get_store()
    _seed(store)
    store.delete_source("alice", "a.txt")
    assert store.list_sources("alice") == []
    assert store.list_sources("bob") == [("b.txt", 1)]


def test_delete_all_for_user_leaves_others_intact():
    store = get_store()
    _seed(store)
    store.delete_all_for_user("alice")
    assert store.list_sources("alice") == []
    assert store.list_sources("bob") == [("b.txt", 1)]


# --- Chat-scoped ingestion ---------------------------------------------------
def _seed_chats(store):
    """Same user, two chats, plus one user-global (no-conversation) doc."""
    store.upsert(["chat one note"], [VEC_A], user_id="alice",
                 source="one.txt", conversation_id="c1")
    store.upsert(["chat two note"], [VEC_B], user_id="alice",
                 source="two.txt", conversation_id="c2")
    store.upsert(["global note"], [VEC_A], user_id="alice", source="g.txt")


def test_search_is_scoped_to_conversation():
    store = get_store()
    _seed_chats(store)
    # Querying chat one (even with chat two's vector) only sees chat one's doc.
    hits = store.search(VEC_B, user_id="alice", top_k=10, conversation_id="c1")
    assert {h.source for h in hits} == {"one.txt"}


def test_list_sources_is_per_conversation():
    store = get_store()
    _seed_chats(store)
    assert store.list_sources("alice", conversation_id="c1") == [("one.txt", 1)]
    assert store.list_sources("alice", conversation_id="c2") == [("two.txt", 1)]


def test_delete_conversation_only_affects_that_chat():
    store = get_store()
    _seed_chats(store)
    store.delete_conversation("alice", "c1")
    assert store.list_sources("alice", conversation_id="c1") == []
    assert store.list_sources("alice", conversation_id="c2") == [("two.txt", 1)]
