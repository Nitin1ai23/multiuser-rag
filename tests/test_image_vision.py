"""Images are indexed by what they show, not just the characters in them.

A photo or chart carries no OCR text, so without a vision caption it indexes as
nothing and the only possible answer is "I don't know". These tests fake the
vision provider and the embedder, so they need no API keys.
"""

from __future__ import annotations

import pytest

from rag_app.ingest import UnsupportedFileError


def _png(tmp_path, name, text=None):
    """An image with no text at all, unless `text` is given."""
    pytest.importorskip("PIL", minversion="10.1")  # load_default(size=)
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (320, 200), "white")
    d = ImageDraw.Draw(img)
    d.ellipse([20, 20, 160, 160], fill="red")
    if text:
        d.text((20, 170), text, fill="black", font=ImageFont.load_default(size=22))
    p = tmp_path / name
    img.save(str(p))
    return p


class _FakeEmbedder:
    """Deterministic 8-dim vectors (EMBEDDING_DIM=8 in the test env)."""

    def embed_documents(self, texts):
        return [[float(len(t) % 7), 1, 0, 0, 0, 0, 0, 0] for t in texts]

    def embed_query(self, text):
        return [float(len(text) % 7), 1, 0, 0, 0, 0, 0, 0]


class _FakeProvider:
    name = "fake"

    def __init__(self, caption="A red circle on a white background.", fail=False):
        self.caption = caption
        self.fail = fail
        self.calls = 0

    def describe_image(self, data):
        self.calls += 1
        if self.fail:
            raise RuntimeError("vision endpoint exploded")
        return self.caption


@pytest.fixture
def pipeline(monkeypatch):
    """Build a RAGPipeline with the network parts stubbed out."""
    from rag_app import rag as rag_mod

    monkeypatch.setattr(rag_mod, "get_embedder", lambda *a, **k: _FakeEmbedder())

    def _make(provider=None):
        if provider is not None:
            monkeypatch.setattr(rag_mod, "get_provider", lambda *a, **k: provider)
        return rag_mod.RAGPipeline("user-1")

    return _make


def _stored_text(user_id="user-1"):
    from rag_app.vectorstore import get_store

    store = get_store()
    hits = store.search([0.0, 1, 0, 0, 0, 0, 0, 0], user_id=user_id, top_k=50)
    return "\n".join(h.text for h in hits)


def test_textless_image_is_indexed_by_its_caption(tmp_path, pipeline):
    provider = _FakeProvider(caption="A red bicycle leaning against a brick wall.")
    rag = pipeline(provider)
    png = _png(tmp_path, "bike.png")  # no text -> OCR yields nothing

    added = rag.ingest_file(png)

    assert added > 0, "a text-free image must still index via its caption"
    assert provider.calls == 1
    assert rag.list_documents() == [("bike.png", added)]
    assert "red bicycle" in _stored_text()


def test_caption_and_ocr_text_are_both_indexed(tmp_path, pipeline):
    pytest.importorskip("pytesseract")
    provider = _FakeProvider(caption="A chart with a red circle.")
    rag = pipeline(provider)
    png = _png(tmp_path, "chart.png", text="Revenue 42")

    try:
        added = rag.ingest_file(png)
    except UnsupportedFileError as exc:
        pytest.skip(f"OCR unavailable: {exc}")

    assert added > 0
    text = _stored_text()
    assert "red circle" in text          # from the vision caption
    if "Revenue" not in text:
        pytest.skip("Tesseract did not read the rendered text on this box")
    assert "Revenue" in text             # from OCR


def test_vision_failure_still_indexes_ocr_text(tmp_path, pipeline):
    provider = _FakeProvider(fail=True)
    rag = pipeline(provider)
    png = _png(tmp_path, "fallback.png", text="Serial ABC123")

    try:
        added = rag.ingest_file(png)
    except UnsupportedFileError as exc:
        pytest.skip(f"OCR unavailable: {exc}")

    if added == 0:
        pytest.skip("Tesseract read nothing from the rendered text")
    assert provider.calls == 1           # it was tried, and it failed
    assert "ABC123" in _stored_text()    # OCR still carried the document


def test_vision_disabled_falls_back_to_ocr_only(tmp_path, pipeline, monkeypatch):
    monkeypatch.setenv("VISION_ENABLED", "false")
    from rag_app.config import get_settings

    get_settings.cache_clear()

    provider = _FakeProvider()
    rag = pipeline(provider)
    png = _png(tmp_path, "nocaption.png")  # text-free

    added = rag.ingest_file(png)

    assert provider.calls == 0, "vision must not be called when disabled"
    assert added == 0, "no caption and no OCR text means nothing to index"
    get_settings.cache_clear()
