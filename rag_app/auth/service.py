"""Authentication service: signup, login, and security-question reset.

Passwords and security answers are stored as salted PBKDF2-HMAC-SHA256 hashes
(never in plaintext). All comparisons use ``hmac.compare_digest`` to avoid
timing leaks. Each user gets a UUID that is used as their data-isolation key in
Qdrant and the chat-history table.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import uuid
from dataclasses import dataclass
from typing import Any

from ..config import Settings, get_settings
from .db import IntegrityError, get_db

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# The preset security questions offered at signup. Kept server-side so the list
# is shared and centrally editable; signup rejects any question not from it.
SECURITY_QUESTIONS: tuple[str, ...] = (
    "What was the name of your first pet?",
    "What was the first school you attended?",
    "What is your mother's maiden name?",
    "In what city were you born?",
    "What was the make of your first car?",
    "What is your favorite book?",
)


class AuthError(Exception):
    """Raised for any signup/login/reset failure with a user-facing message."""


@dataclass(frozen=True)
class User:
    id: str
    username: str
    email: str


def _hash(secret: str, salt: bytes, iterations: int) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", secret.encode("utf-8"), salt, iterations)
    return dk.hex()


def _new_salt() -> bytes:
    return os.urandom(16)


class AuthService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.db = get_db()
        self.iterations = self.settings.pbkdf2_iterations

    # --- Signup ----------------------------------------------------------
    def signup(
        self,
        username: str,
        email: str,
        password: str,
        security_question: str,
        security_answer: str,
    ) -> User:
        username = (username or "").strip()
        email = (email or "").strip().lower()
        security_question = (security_question or "").strip()
        answer = (security_answer or "").strip()

        if len(username) < 3:
            raise AuthError("Username must be at least 3 characters.")
        if not _EMAIL_RE.match(email):
            raise AuthError("Please enter a valid email address.")
        if len(password) < 8:
            raise AuthError("Password must be at least 8 characters.")
        if not security_question:
            raise AuthError("Please choose a security question.")
        if security_question not in SECURITY_QUESTIONS:
            raise AuthError("Please choose one of the offered security questions.")
        if not answer:
            raise AuthError("Please provide a security answer.")

        pwd_salt = _new_salt()
        ans_salt = _new_salt()
        user_id = str(uuid.uuid4())
        try:
            self.db.execute(
                """INSERT INTO users
                   (id, username, email, password_hash, password_salt, iterations,
                    security_question, security_answer_hash, security_answer_salt)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    user_id,
                    username,
                    email,
                    _hash(password, pwd_salt, self.iterations),
                    pwd_salt.hex(),
                    self.iterations,
                    security_question,
                    # answer is normalised to lower-case so casing doesn't matter
                    _hash(answer.lower(), ans_salt, self.iterations),
                    ans_salt.hex(),
                ),
            )
            self.db.commit()
        except IntegrityError:
            # Postgres aborts the transaction on a constraint violation; clear it
            # so the shared connection is usable for the next request.
            self.db.rollback()
            raise AuthError("That username or email is already registered.")
        return User(id=user_id, username=username, email=email)

    # --- Login -----------------------------------------------------------
    def login(self, identifier: str, password: str) -> User:
        """Authenticate by username OR email plus password."""
        identifier = (identifier or "").strip()
        row = self._find_user(identifier)
        # Always run a hash to keep timing uniform whether or not the user exists.
        if row is None:
            _hash(password, b"dummy-salt-padding", self.iterations)
            raise AuthError("Invalid credentials.")
        expected = row["password_hash"]
        candidate = _hash(
            password, bytes.fromhex(row["password_salt"]), row["iterations"]
        )
        if not hmac.compare_digest(expected, candidate):
            raise AuthError("Invalid credentials.")
        return User(id=row["id"], username=row["username"], email=row["email"])

    # --- Password reset via security question ----------------------------
    def get_security_question(self, identifier: str) -> str:
        row = self._find_user((identifier or "").strip())
        if row is None:
            raise AuthError("No account found for that username or email.")
        return row["security_question"]

    def reset_password(
        self, identifier: str, security_answer: str, new_password: str
    ) -> User:
        identifier = (identifier or "").strip()
        answer = (security_answer or "").strip().lower()
        row = self._find_user(identifier)
        if row is None:
            raise AuthError("No account found for that username or email.")
        candidate = _hash(
            answer, bytes.fromhex(row["security_answer_salt"]), row["iterations"]
        )
        if not hmac.compare_digest(row["security_answer_hash"], candidate):
            raise AuthError("Security answer is incorrect.")
        if len(new_password) < 8:
            raise AuthError("New password must be at least 8 characters.")

        new_salt = _new_salt()
        self.db.execute(
            "UPDATE users SET password_hash = ?, password_salt = ?, iterations = ? "
            "WHERE id = ?",
            (
                _hash(new_password, new_salt, self.iterations),
                new_salt.hex(),
                self.iterations,
                row["id"],
            ),
        )
        self.db.commit()
        return User(id=row["id"], username=row["username"], email=row["email"])

    # --- Account deletion ------------------------------------------------
    def delete_account(self, user_id: str, password: str) -> None:
        """Verify the password, then delete the account and all its rows.

        Conversations and messages cascade via foreign keys. The caller is
        responsible for deleting the user's vectors (the auth layer doesn't
        depend on the vector store).
        """
        row = self.db.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if row is None:
            raise AuthError("Account not found.")
        candidate = _hash(
            password, bytes.fromhex(row["password_salt"]), row["iterations"]
        )
        if not hmac.compare_digest(row["password_hash"], candidate):
            raise AuthError("Password is incorrect.")
        # Explicitly clear child rows in case foreign keys aren't enforced on
        # an older, migrated database.
        self.db.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
        self.db.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
        self.db.execute("DELETE FROM users WHERE id = ?", (user_id,))
        self.db.commit()

    # --- Helpers ---------------------------------------------------------
    def _find_user(self, identifier: str) -> dict[str, Any] | None:
        return self.db.execute(
            "SELECT * FROM users WHERE username = ? OR email = ?",
            (identifier, identifier.lower()),
        ).fetchone()
