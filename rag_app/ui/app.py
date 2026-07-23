"""Application bootstrap and window orchestration.

Shows the login window; on successful authentication, opens a main window bound
to that user. On logout, returns to the login window. Only one user's windows
exist at a time.
"""

from __future__ import annotations

import sys

from PyQt5.QtWidgets import QApplication

from ..auth.service import AuthService, User
from .login_window import LoginWindow
from .main_window import MainWindow


class AppController:
    def __init__(self) -> None:
        self.auth = AuthService()
        self.login_window: LoginWindow | None = None
        self.main_window: MainWindow | None = None

    def start(self) -> None:
        self._show_login()

    def _show_login(self) -> None:
        self.login_window = LoginWindow(self.auth)
        self.login_window.authenticated.connect(self._on_authenticated)
        self.login_window.show()

    def _on_authenticated(self, user: User) -> None:
        if self.login_window is not None:
            self.login_window.close()
            self.login_window = None
        self.main_window = MainWindow(user)
        self.main_window.logout.connect(self._on_logout)
        self.main_window.show()

    def _on_logout(self) -> None:
        if self.main_window is not None:
            self.main_window.close()
            self.main_window = None
        self._show_login()


def run() -> int:
    app = QApplication(sys.argv)
    controller = AppController()
    controller.start()
    return app.exec_()
