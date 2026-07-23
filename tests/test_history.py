from rag_app.rag import _normalize_history


class _Msg:
    def __init__(self, role, content):
        self.role = role
        self.content = content


def test_normalizes_tuples_objects_and_none():
    assert _normalize_history(None) == []
    assert _normalize_history([("user", "hi")]) == [("user", "hi")]
    assert _normalize_history([_Msg("assistant", "yo")]) == [("assistant", "yo")]
    mixed = _normalize_history([("user", "a"), _Msg("assistant", "b")])
    assert mixed == [("user", "a"), ("assistant", "b")]
