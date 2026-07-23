"""Document loading and chunking.

Loading turns an uploaded file into one or more ``(source, text)`` documents:

* **Text & code** — any plain-text file (prose, markdown, source code in any
  language, CSV/TSV, JSON/YAML/TOML, logs, config) is read directly as UTF-8.
* **HTML/XML** — tags are stripped to readable text.
* **PDF / Word / Excel / PowerPoint** — parsed via dedicated libraries.
* **Images** — OCR'd (needs the Tesseract binary at runtime). OCR only recovers
  characters; captioning an image's visual content is the RAG layer's job (see
  ``RAGPipeline._ingest_image``), which keeps this module offline and cheap.
* **Archives (.zip)** — expanded recursively; every readable member becomes its
  own document, named ``archive.zip/path/inside.ext`` so citations stay precise.

Chunking is character-based with overlap and tries to break on
paragraph/sentence boundaries for cleaner splits.
"""

from __future__ import annotations

import importlib
import io
import zipfile
from pathlib import Path

from .config import Settings, get_settings

# --- Supported file types ---------------------------------------------------
# Read verbatim as UTF-8: prose, markup, data, config, and source code. This is
# deliberately broad ("everything from coding to texts"); anything not matched
# by a richer parser below falls through to here if it's in this set.
_TEXT_SUFFIXES = frozenset({
    # prose / markup / docs
    ".txt", ".text", ".md", ".markdown", ".mdx", ".rst", ".log", ".tex",
    # data / config
    ".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".conf", ".env", ".properties", ".editorconfig",
    # web (non-tag styling/text)
    ".css", ".scss", ".sass", ".less",
    # source code
    ".py", ".pyi", ".ipynb", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx",
    ".java", ".kt", ".kts", ".scala", ".groovy", ".c", ".h", ".cc", ".cpp",
    ".cxx", ".hpp", ".hh", ".cs", ".go", ".rs", ".rb", ".php", ".swift",
    ".m", ".mm", ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd",
    ".sql", ".r", ".jl", ".lua", ".pl", ".pm", ".dart", ".ex", ".exs",
    ".erl", ".clj", ".cljs", ".hs", ".ml", ".fs", ".vb", ".asm", ".vim",
    ".gradle", ".cmake", ".mk", ".dockerfile", ".gitignore", ".gitattributes",
})
_HTML_SUFFIXES = frozenset({".html", ".htm", ".xhtml", ".xml", ".svg"})
_PDF_SUFFIXES = frozenset({".pdf"})
_DOCX_SUFFIXES = frozenset({".docx"})
_EXCEL_SUFFIXES = frozenset({".xlsx", ".xlsm"})
_PPTX_SUFFIXES = frozenset({".pptx"})
IMAGE_SUFFIXES = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp",
})
_ARCHIVE_SUFFIXES = frozenset({".zip"})

# Everything the uploader will accept. Archives are included even though they
# hold their own documents, so the size-checked upload path lets them through.
SUPPORTED_SUFFIXES = frozenset(
    _TEXT_SUFFIXES
    | _HTML_SUFFIXES
    | _PDF_SUFFIXES
    | _DOCX_SUFFIXES
    | _EXCEL_SUFFIXES
    | _PPTX_SUFFIXES
    | IMAGE_SUFFIXES
    | _ARCHIVE_SUFFIXES
)

# Guards against zip bombs / runaway archives.
_MAX_ARCHIVE_DEPTH = 3          # nested-zip recursion limit
_MAX_MEMBER_BYTES = 50 * 1024 * 1024  # skip any single member larger than this
_MAX_ARCHIVE_MEMBERS = 2000     # cap documents pulled from one archive


class UnsupportedFileError(ValueError):
    """Raised for a file type we don't know how to read."""


def _require(module: str, package: str, purpose: str):
    """Import an optional parser lib, or raise a clear, actionable message."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on env
        raise UnsupportedFileError(
            f"Cannot read {purpose}: the '{package}' library is not installed."
        ) from exc


# --- Per-format extractors (operate on raw bytes) ---------------------------
def _extract_pdf(data: bytes) -> str:
    PdfReader = _require("pypdf", "pypdf", "PDF files").PdfReader
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _extract_docx(data: bytes) -> str:
    docx = _require("docx", "python-docx", "Word documents")
    doc = docx.Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append("\t".join(cells))
    return "\n".join(parts)


def _extract_xlsx(data: bytes) -> str:
    openpyxl = _require("openpyxl", "openpyxl", "Excel spreadsheets")
    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    parts: list[str] = []
    for ws in wb.worksheets:
        parts.append(f"# Sheet: {ws.title}")
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) for c in row if c is not None]
            if cells:
                parts.append("\t".join(cells))
    wb.close()
    return "\n".join(parts)


def _extract_pptx(data: bytes) -> str:
    pptx = _require("pptx", "python-pptx", "PowerPoint presentations")
    prs = pptx.Presentation(io.BytesIO(data))
    parts: list[str] = []
    for i, slide in enumerate(prs.slides, start=1):
        parts.append(f"# Slide {i}")
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                parts.append(shape.text_frame.text)
    return "\n".join(parts)


def _extract_image(data: bytes, name: str) -> str:
    """OCR an image. Needs the Tesseract binary; explains itself if it's absent."""
    Image = _require("PIL.Image", "Pillow", "images")
    pytesseract = _require("pytesseract", "pytesseract", "images")
    try:
        with Image.open(io.BytesIO(data)) as img:
            return pytesseract.image_to_string(img)
    except pytesseract.TesseractNotFoundError as exc:
        raise UnsupportedFileError(
            "Cannot read images: the Tesseract OCR engine is not installed "
            "(install it and ensure `tesseract` is on PATH)."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - bad/unreadable image
        raise UnsupportedFileError(f"Could not read image {name}: {exc}") from exc


def _strip_html(markup: str) -> str:
    """Reduce HTML/XML to readable text (no external dependency)."""
    from html.parser import HTMLParser

    class _Text(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.out: list[str] = []
            self._skip = 0

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style"):
                self._skip += 1

        def handle_endtag(self, tag):
            if tag in ("script", "style") and self._skip:
                self._skip -= 1

        def handle_data(self, data):
            if not self._skip and data.strip():
                self.out.append(data)

    parser = _Text()
    parser.feed(markup)
    return "\n".join(parser.out)


def _extract(suffix: str, data: bytes, name: str) -> str:
    """Turn one file's bytes into text, dispatching on its extension."""
    if suffix in _TEXT_SUFFIXES:
        return data.decode("utf-8", errors="ignore")
    if suffix in _HTML_SUFFIXES:
        return _strip_html(data.decode("utf-8", errors="ignore"))
    if suffix in _PDF_SUFFIXES:
        return _extract_pdf(data)
    if suffix in _DOCX_SUFFIXES:
        return _extract_docx(data)
    if suffix in _EXCEL_SUFFIXES:
        return _extract_xlsx(data)
    if suffix in _PPTX_SUFFIXES:
        return _extract_pptx(data)
    if suffix in IMAGE_SUFFIXES:
        return _extract_image(data, name)
    raise UnsupportedFileError(f"Unsupported file type: {suffix or '(none)'} ({name})")


# --- Archives ---------------------------------------------------------------
def _skip_member(name: str) -> bool:
    """Ignore OS/junk entries that carry no useful content."""
    base = name.rsplit("/", 1)[-1]
    return "__MACOSX" in name or base in {".DS_Store", "Thumbs.db"}


def _iter_archive(fileobj, prefix: str, depth: int, out: list[tuple[str, str]]) -> None:
    if depth > _MAX_ARCHIVE_DEPTH:
        return
    try:
        zf = zipfile.ZipFile(fileobj)
    except zipfile.BadZipFile:
        return
    with zf:
        for info in zf.infolist():
            if len(out) >= _MAX_ARCHIVE_MEMBERS:
                break
            if info.is_dir() or _skip_member(info.filename):
                continue
            if info.file_size > _MAX_MEMBER_BYTES:
                continue
            suffix = Path(info.filename).suffix.lower()
            source = f"{prefix}/{info.filename}"
            if suffix in _ARCHIVE_SUFFIXES:
                _iter_archive(io.BytesIO(zf.read(info)), source, depth + 1, out)
                continue
            if suffix not in SUPPORTED_SUFFIXES:
                continue  # unknown member type — skip, don't fail the whole zip
            try:
                text = _extract(suffix, zf.read(info), info.filename)
            except Exception:  # noqa: BLE001 - skip an unreadable member, keep the rest
                continue
            if text.strip():
                out.append((source, text))


# --- Public API -------------------------------------------------------------
def load_documents(path: str | Path) -> list[tuple[str, str]]:
    """Load a file into one or more ``(source, text)`` documents.

    A normal file yields a single document named after the file; an archive
    yields one document per readable member.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in _ARCHIVE_SUFFIXES:
        out: list[tuple[str, str]] = []
        _iter_archive(str(path), path.name, 0, out)
        return out
    text = load_text(path)
    return [(path.name, text)] if text.strip() else []


def load_text(path: str | Path) -> str:
    """Extract raw text from a single supported (non-archive) file."""
    path = Path(path)
    return _extract(path.suffix.lower(), path.read_bytes(), path.name)


def chunk_text(text: str, settings: Settings | None = None) -> list[str]:
    """Split text into overlapping chunks, preferring natural boundaries."""
    settings = settings or get_settings()
    size = settings.chunk_size
    overlap = settings.chunk_overlap

    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            # Try to break on a paragraph, then sentence, then whitespace.
            window = text[start:end]
            for sep in ("\n\n", "\n", ". ", " "):
                idx = window.rfind(sep)
                if idx > size * 0.5:  # only break if reasonably far in
                    end = start + idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_and_chunk(path: str | Path, settings: Settings | None = None) -> list[str]:
    """Convenience: load a single (non-archive) file and return its chunks."""
    return chunk_text(load_text(path), settings)
