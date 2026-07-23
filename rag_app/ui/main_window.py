"""Per-user main window: chat with your documents + manage your documents.

Constructed with a ``User``; it builds a ``RAGPipeline`` and ``ChatHistory``
bound to that user's id, so all data shown and stored belongs only to them.
Emits ``logout`` to return to the sign-in screen.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..auth.service import User
from ..chat.history import ChatHistory
from .workers import IngestWorker, InitWorker, QueryWorker


class MainWindow(QMainWindow):
    logout = pyqtSignal()

    def __init__(self, user: User) -> None:
        super().__init__()
        self.user = user
        self.pipeline = None
        self.history = ChatHistory(user.id)
        self.history.clear()
        self._workers: list = []  # keep references so threads aren't GC'd

        self.setWindowTitle(f"RAG Assistant — {user.username}")
        self.resize(900, 640)
        self._build_ui()
        self._set_busy(True, "Initializing…")
        self._init_pipeline()

    # --------------------------------------------------------------- UI build
    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)

        header = QHBoxLayout()
        header.addWidget(QLabel(f"Signed in as <b>{self.user.username}</b>"))
        header.addStretch()
        logout_btn = QPushButton("Log out")
        logout_btn.clicked.connect(self._on_logout)
        header.addWidget(logout_btn)
        root.addLayout(header)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_chat_tab(), "Chat")
        self.tabs.addTab(self._build_docs_tab(), "Documents")
        root.addWidget(self.tabs)

        self.status = QLabel("")
        self.status.setStyleSheet("color: #666;")
        root.addWidget(self.status)

        self.setCentralWidget(central)

    def _build_chat_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.transcript = QTextBrowser()
        self.transcript.setOpenExternalLinks(False)
        layout.addWidget(self.transcript, stretch=1)

        row = QHBoxLayout()
        self.question = QTextEdit()
        self.question.setPlaceholderText("Ask a question about your documents…")
        self.question.setFixedHeight(70)
        row.addWidget(self.question, stretch=1)
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self._on_send)
        row.addWidget(self.send_btn)
        layout.addLayout(row)

        clear = QPushButton("Clear chat history")
        clear.clicked.connect(self._on_clear_history)
        layout.addWidget(clear, alignment=Qt.AlignLeft)

        self._render_history()
        return tab

    def _build_docs_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("Your documents (only you can see these):"))
        self.doc_list = QListWidget()
        layout.addWidget(self.doc_list, stretch=1)

        row = QHBoxLayout()
        upload = QPushButton("Upload & ingest…")
        upload.clicked.connect(self._on_upload)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._refresh_docs)
        delete = QPushButton("Delete selected")
        delete.clicked.connect(self._on_delete_doc)
        row.addWidget(upload)
        row.addWidget(refresh)
        row.addWidget(delete)
        row.addStretch()
        layout.addLayout(row)
        return tab

    # ----------------------------------------------------------- pipeline init
    def _init_pipeline(self) -> None:
        worker = InitWorker(self.user.id)
        worker.finished.connect(self._on_pipeline_ready)
        worker.error.connect(self._on_pipeline_error)
        self._track(worker)
        worker.start()

    def _on_pipeline_ready(self, pipeline) -> None:
        self.pipeline = pipeline
        self._set_busy(False, "Ready.")
        self._refresh_docs()

    def _on_pipeline_error(self, message: str) -> None:
        self._set_busy(False, "Initialization failed.")
        QMessageBox.critical(
            self,
            "Setup error",
            f"Could not start the RAG pipeline:\n\n{message}\n\n"
            "Check your NVIDIA_API_KEY and GROQ_API_KEY in the .env file.",
        )

    # ------------------------------------------------------------------- chat
    def _render_history(self) -> None:
        self.transcript.clear()
        for msg in self.history.all():
            self._append_bubble(msg.role, msg.content)

    def _append_bubble(self, role: str, content: str) -> None:
        who = "You" if role == "user" else "Assistant"
        color = "#1a73e8" if role == "user" else "#0b8043"
        safe = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        safe = safe.replace("\n", "<br>")
        self.transcript.append(
            f'<p><b style="color:{color}">{who}:</b><br>{safe}</p>'
        )

    def _on_send(self) -> None:
        if self.pipeline is None:
            QMessageBox.information(self, "Please wait", "Still initializing…")
            return
        question = self.question.toPlainText().strip()
        if not question:
            return
        self.question.clear()
        prior = self.history.all()  # turns before this question, for context
        self.history.add("user", question)
        self._append_bubble("user", question)
        self._set_busy(True, "Thinking…")

        worker = QueryWorker(self.pipeline, question, history=prior)
        worker.finished.connect(self._on_answer)
        worker.error.connect(self._on_query_error)
        self._track(worker)
        worker.start()

    def _on_answer(self, answer) -> None:
        self._set_busy(False, f"Answered via {answer.provider}.")
        text = answer.answer
        if answer.sources:
            srcs = ", ".join(sorted({c.source for c in answer.sources}))
            text += f"\n\n— sources: {srcs}"
        self.history.add("assistant", text)
        self._append_bubble("assistant", text)

    def _on_query_error(self, message: str) -> None:
        self._set_busy(False, "Query failed.")
        QMessageBox.warning(self, "Query error", message)

    def _on_clear_history(self) -> None:
        if (
            QMessageBox.question(self, "Clear chat", "Delete your chat history?")
            == QMessageBox.Yes
        ):
            self.history.clear()
            self.transcript.clear()

    # -------------------------------------------------------------- documents
    def _on_upload(self) -> None:
        if self.pipeline is None:
            QMessageBox.information(self, "Please wait", "Still initializing…")
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select documents",
            "",
            "Documents (*.pdf *.txt *.md *.markdown *.rst);;All files (*)",
        )
        if not paths:
            return
        self._set_busy(True, f"Ingesting {len(paths)} file(s)…")
        worker = IngestWorker(self.pipeline, paths)
        worker.file_done.connect(
            lambda name, n: self.status.setText(f"Ingested {name}: {n} chunks")
        )
        worker.file_error.connect(
            lambda name, err: QMessageBox.warning(self, "Ingest error", f"{name}\n{err}")
        )
        worker.all_done.connect(self._on_ingest_done)
        self._track(worker)
        worker.start()

    def _on_ingest_done(self, total: int) -> None:
        self._set_busy(False, f"Done. {total} chunks added.")
        self._refresh_docs()

    def _refresh_docs(self) -> None:
        if self.pipeline is None:
            return
        self.doc_list.clear()
        try:
            for source, count in self.pipeline.list_documents():
                self.doc_list.addItem(f"{source}  ({count} chunks)")
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f"Could not list documents: {exc}")

    def _on_delete_doc(self) -> None:
        item = self.doc_list.currentItem()
        if item is None or self.pipeline is None:
            return
        source = item.text().rsplit("  (", 1)[0]
        if (
            QMessageBox.question(self, "Delete", f"Delete '{source}'?")
            != QMessageBox.Yes
        ):
            return
        self.pipeline.delete_document(source)
        self._refresh_docs()

    # ----------------------------------------------------------------- common
    def _on_logout(self) -> None:
        self.logout.emit()

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self.send_btn.setEnabled(not busy)
        if message:
            self.status.setText(message)

    def _track(self, worker) -> None:
        self._workers.append(worker)
        worker.finished.connect(lambda *_: self._workers.remove(worker)
                                if worker in self._workers else None)
