from rag_app.config import get_settings
from rag_app.ingest import chunk_text


def test_empty_text_yields_no_chunks():
    assert chunk_text("   \n  ") == []


def test_short_text_is_a_single_chunk():
    assert chunk_text("hello world") == ["hello world"]


def test_long_text_splits_with_bounded_size():
    settings = get_settings()  # CHUNK_SIZE=100 in the test env
    text = "Sentence number %d. " % 0 + "".join(
        "Sentence number %d. " % i for i in range(1, 200)
    )
    chunks = chunk_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= settings.chunk_size for c in chunks)


def test_chunks_overlap_for_continuity():
    text = "".join(f"word{i} " for i in range(400))
    chunks = chunk_text(text)
    # Consecutive chunks should share some trailing/leading text (overlap > 0).
    assert len(chunks) >= 2
    tail = chunks[0][-10:]
    assert tail.strip() and tail in chunks[0]
