"""Loading a spread of file types into (source, text) documents."""

import io
import zipfile

import pytest

from rag_app.ingest import (
    SUPPORTED_SUFFIXES,
    UnsupportedFileError,
    load_documents,
    load_text,
)


def _write(tmp_path, name, data):
    p = tmp_path / name
    p.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    return p


def test_plain_text_and_code_are_read_verbatim(tmp_path):
    py = _write(tmp_path, "hello.py", "def add(a, b):\n    return a + b\n")
    assert "return a + b" in load_text(py)


def test_csv_and_json_are_supported(tmp_path):
    csv = _write(tmp_path, "data.csv", "name,age\nAlice,30\nBob,25\n")
    js = _write(tmp_path, "cfg.json", '{"key": "value", "n": 1}')
    assert "Alice" in load_text(csv)
    assert "value" in load_text(js)


def test_html_tags_are_stripped(tmp_path):
    html = _write(
        tmp_path,
        "page.html",
        "<html><head><style>b{color:red}</style></head>"
        "<body><h1>Title</h1><p>Body text.</p></body></html>",
    )
    text = load_text(html)
    assert "Title" in text and "Body text." in text
    assert "color:red" not in text  # <style> content dropped


def test_docx_paragraphs_are_extracted(tmp_path):
    docx = pytest.importorskip("docx")
    doc = docx.Document()
    doc.add_paragraph("First paragraph.")
    doc.add_paragraph("Second paragraph.")
    p = tmp_path / "doc.docx"
    doc.save(str(p))
    text = load_text(p)
    assert "First paragraph." in text and "Second paragraph." in text


def test_xlsx_cells_are_extracted(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["City", "Pop"])
    ws.append(["Paris", 2000000])
    p = tmp_path / "book.xlsx"
    wb.save(str(p))
    text = load_text(p)
    assert "Paris" in text and "City" in text


def _render_text_png(tmp_path, name, text):
    """Render ``text`` to a PNG legible enough for OCR, or skip if we can't."""
    pytest.importorskip("PIL", minversion="10.1")  # load_default(size=)
    from PIL import Image, ImageDraw, ImageFont

    pytesseract = pytest.importorskip("pytesseract")
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        pytest.skip("Tesseract binary not on PATH")

    img = Image.new("RGB", (760, 160), "white")
    ImageDraw.Draw(img).text(
        (20, 40), text, fill="black", font=ImageFont.load_default(size=48)
    )
    p = tmp_path / name
    img.save(str(p))
    return p


def test_image_text_is_ocred(tmp_path):
    png = _render_text_png(tmp_path, "note.png", "Ocean sunfish")
    assert "Ocean sunfish" in load_text(png)


def test_image_without_text_yields_no_documents(tmp_path):
    pytest.importorskip("pytesseract")
    Image = pytest.importorskip("PIL.Image")
    p = tmp_path / "blank.png"
    Image.new("RGB", (200, 200), "white").save(str(p))
    assert load_documents(p) == []


def test_unreadable_image_raises(tmp_path):
    pytest.importorskip("pytesseract")
    pytest.importorskip("PIL.Image")
    bad = _write(tmp_path, "bad.png", b"not an image at all")
    with pytest.raises(UnsupportedFileError):
        load_text(bad)


def test_zip_expands_into_one_document_per_member(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("a.txt", "alpha content")
        zf.writestr("src/b.py", "print('beta')")
        zf.writestr("__MACOSX/junk", "ignore me")   # skipped
        zf.writestr("logo.bin", b"\x00\x01\x02")     # unsupported — skipped
    p = _write(tmp_path, "bundle.zip", buf.getvalue())

    docs = dict(load_documents(p))
    assert docs == {
        "bundle.zip/a.txt": "alpha content",
        "bundle.zip/src/b.py": "print('beta')",
    }


def test_unsupported_type_raises(tmp_path):
    weird = _write(tmp_path, "mystery.xyz", b"\x00\x01")
    with pytest.raises(UnsupportedFileError):
        load_text(weird)


def test_supported_set_covers_the_common_types():
    for ext in (".py", ".txt", ".csv", ".json", ".pdf", ".docx", ".png", ".zip"):
        assert ext in SUPPORTED_SUFFIXES
