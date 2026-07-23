"""Background QThread workers.

Network / disk work (building the pipeline, embedding, ingesting, querying Groq)
runs off the GUI thread so the window stays responsive.
"""

from __future__ import annotations

from PyQt5.QtCore import QThread, pyqtSignal

from ..rag import Answer, RAGPipeline


class InitWorker(QThread):
    """Construct the per-user RAGPipeline in the background."""

    finished = pyqtSignal(object)  # RAGPipeline
    error = pyqtSignal(str)

    def __init__(self, user_id: str) -> None:
        super().__init__()
        self.user_id = user_id

    def run(self) -> None:
        try:
            self.finished.emit(RAGPipeline(self.user_id))
        except Exception as exc:  # noqa: BLE001 - surface any setup failure
            self.error.emit(str(exc))


class IngestWorker(QThread):
    """Ingest files one-by-one, reporting progress per file."""

    file_done = pyqtSignal(str, int)   # (filename, chunk_count)
    file_error = pyqtSignal(str, str)  # (filename, error_message)
    all_done = pyqtSignal(int)         # total_chunks

    def __init__(self, pipeline: RAGPipeline, paths: list[str]) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.paths = paths

    def run(self) -> None:
        total = 0
        for path in self.paths:
            try:
                count = self.pipeline.ingest_file(path)
                total += count
                self.file_done.emit(path, count)
            except Exception as exc:  # noqa: BLE001
                self.file_error.emit(path, str(exc))
        self.all_done.emit(total)


class QueryWorker(QThread):
    """Run a single RAG query (condense -> embed -> retrieve -> rerank -> Groq)."""

    finished = pyqtSignal(object)  # Answer
    error = pyqtSignal(str)

    def __init__(
        self, pipeline: RAGPipeline, question: str, history: list | None = None
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.question = question
        self.history = history or []  # prior turns, for follow-up understanding

    def run(self) -> None:
        try:
            answer: Answer = self.pipeline.query(self.question, history=self.history)
            self.finished.emit(answer)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
