"""Authentication window: login, sign up, and forgot-password screens.

Emits ``authenticated(User)`` once a user logs in (or signs up). The app then
swaps this window out for the per-user main window.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..auth.service import AuthError, AuthService

SECURITY_QUESTIONS = [
    "What was the name of your first pet?",
    "What is your mother's maiden name?",
    "What city were you born in?",
    "What was the name of your first school?",
    "What is your favorite book?",
]


class LoginWindow(QWidget):
    authenticated = pyqtSignal(object)  # User

    def __init__(self, auth: AuthService | None = None) -> None:
        super().__init__()
        self.auth = auth or AuthService()
        self.setWindowTitle("RAG Assistant — Sign in")
        self.resize(420, 460)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_login_page())     # index 0
        self.stack.addWidget(self._build_signup_page())    # index 1
        self.stack.addWidget(self._build_forgot_page())    # index 2

        root = QVBoxLayout(self)
        title = QLabel("RAG Assistant")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: 600; margin: 12px;")
        root.addWidget(title)
        root.addWidget(self.stack)

    # ------------------------------------------------------------------ login
    def _build_login_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        form = QFormLayout()
        self.login_id = QLineEdit()
        self.login_id.setPlaceholderText("username or email")
        self.login_pw = QLineEdit()
        self.login_pw.setEchoMode(QLineEdit.Password)
        self.login_pw.setPlaceholderText("password")
        self.login_pw.returnPressed.connect(self._do_login)
        form.addRow("Username / Email", self.login_id)
        form.addRow("Password", self.login_pw)
        layout.addLayout(form)

        sign_in = QPushButton("Sign in")
        sign_in.clicked.connect(self._do_login)
        layout.addWidget(sign_in)

        links = QHBoxLayout()
        create = QPushButton("Create account")
        create.setFlat(True)
        create.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        forgot = QPushButton("Forgot password?")
        forgot.setFlat(True)
        forgot.clicked.connect(lambda: self.stack.setCurrentIndex(2))
        links.addWidget(create)
        links.addStretch()
        links.addWidget(forgot)
        layout.addLayout(links)
        layout.addStretch()
        return page

    def _do_login(self) -> None:
        try:
            user = self.auth.login(self.login_id.text(), self.login_pw.text())
        except AuthError as exc:
            QMessageBox.warning(self, "Sign in failed", str(exc))
            return
        self.login_pw.clear()
        self.authenticated.emit(user)

    # ----------------------------------------------------------------- signup
    def _build_signup_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        form = QFormLayout()
        self.su_user = QLineEdit()
        self.su_email = QLineEdit()
        self.su_pw = QLineEdit()
        self.su_pw.setEchoMode(QLineEdit.Password)
        self.su_pw2 = QLineEdit()
        self.su_pw2.setEchoMode(QLineEdit.Password)
        self.su_question = QComboBox()
        self.su_question.addItems(SECURITY_QUESTIONS)
        self.su_answer = QLineEdit()

        form.addRow("Username", self.su_user)
        form.addRow("Email", self.su_email)
        form.addRow("Password", self.su_pw)
        form.addRow("Confirm password", self.su_pw2)
        form.addRow("Security question", self.su_question)
        form.addRow("Security answer", self.su_answer)
        layout.addLayout(form)

        create = QPushButton("Create account")
        create.clicked.connect(self._do_signup)
        layout.addWidget(create)

        back = QPushButton("Back to sign in")
        back.setFlat(True)
        back.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        layout.addWidget(back)
        layout.addStretch()
        return page

    def _do_signup(self) -> None:
        if self.su_pw.text() != self.su_pw2.text():
            QMessageBox.warning(self, "Sign up failed", "Passwords do not match.")
            return
        try:
            user = self.auth.signup(
                self.su_user.text(),
                self.su_email.text(),
                self.su_pw.text(),
                self.su_question.currentText(),
                self.su_answer.text(),
            )
        except AuthError as exc:
            QMessageBox.warning(self, "Sign up failed", str(exc))
            return
        QMessageBox.information(
            self, "Welcome", f"Account created. Signed in as {user.username}."
        )
        self.su_pw.clear()
        self.su_pw2.clear()
        self.authenticated.emit(user)

    # ----------------------------------------------------------------- forgot
    def _build_forgot_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        form = QFormLayout()
        self.fp_id = QLineEdit()
        self.fp_id.setPlaceholderText("username or email")
        self.fp_question = QLabel("—")
        self.fp_question.setWordWrap(True)
        self.fp_answer = QLineEdit()
        self.fp_new = QLineEdit()
        self.fp_new.setEchoMode(QLineEdit.Password)
        form.addRow("Username / Email", self.fp_id)
        load = QPushButton("Load security question")
        load.clicked.connect(self._load_question)
        form.addRow("", load)
        form.addRow("Question", self.fp_question)
        form.addRow("Answer", self.fp_answer)
        form.addRow("New password", self.fp_new)
        layout.addLayout(form)

        reset = QPushButton("Reset password")
        reset.clicked.connect(self._do_reset)
        layout.addWidget(reset)

        back = QPushButton("Back to sign in")
        back.setFlat(True)
        back.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        layout.addWidget(back)
        layout.addStretch()
        return page

    def _load_question(self) -> None:
        try:
            question = self.auth.get_security_question(self.fp_id.text())
        except AuthError as exc:
            QMessageBox.warning(self, "Not found", str(exc))
            self.fp_question.setText("—")
            return
        self.fp_question.setText(question)

    def _do_reset(self) -> None:
        try:
            user = self.auth.reset_password(
                self.fp_id.text(), self.fp_answer.text(), self.fp_new.text()
            )
        except AuthError as exc:
            QMessageBox.warning(self, "Reset failed", str(exc))
            return
        QMessageBox.information(
            self, "Password reset", "Your password has been updated. Please sign in."
        )
        self.fp_answer.clear()
        self.fp_new.clear()
        self.stack.setCurrentIndex(0)
        self.login_id.setText(user.username)
