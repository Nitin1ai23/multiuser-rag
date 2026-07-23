"""JWT session tokens and the ``current_user`` dependency.

Login returns a signed JWT carrying the user's id/username/email plus a unique
``jti`` (token id). Every protected endpoint depends on :func:`get_current_user`,
which verifies the token, checks it hasn't been revoked, and reconstructs the
:class:`~rag_app.auth.service.User`. The user id in that token is the same
isolation key used by Qdrant and the chat-history table, so a valid token only
ever grants access to that one user's data.

Logout revokes a token by storing its ``jti`` in a denylist until it expires,
which makes stateless JWTs invalidatable.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..auth.db import get_db
from ..auth.service import User
from ..config import Settings, get_settings

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=True)


def _check_secret(settings: Settings) -> None:
    if settings.jwt_secret == "dev-insecure-change-me":
        logger.warning(
            "JWT_SECRET is the insecure default. Set a strong JWT_SECRET in "
            ".env before exposing this server to anyone."
        )


# --- Revocation denylist -------------------------------------------------
def revoke_jti(jti: str, exp: int) -> None:
    """Add a token id to the denylist until its expiry, and prune stale rows."""
    db = get_db()
    db.execute("DELETE FROM revoked_tokens WHERE exp < ?", (int(time.time()),))
    db.execute(
        "INSERT INTO revoked_tokens (jti, exp) VALUES (?, ?) "
        "ON CONFLICT (jti) DO NOTHING",
        (jti, exp),
    )
    db.commit()


def is_revoked(jti: str) -> bool:
    row = get_db().execute(
        "SELECT 1 FROM revoked_tokens WHERE jti = ?", (jti,)
    ).fetchone()
    return row is not None


# --- Token creation / verification --------------------------------------
def create_access_token(user: User, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    _check_secret(settings)
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.access_token_expire_minutes
    )
    payload = {
        "sub": user.id,
        "username": user.username,
        "email": user.email,
        "jti": uuid.uuid4().hex,
        "exp": expire,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def _decode(
    credentials: HTTPAuthorizationCredentials, settings: Settings
) -> dict:
    invalid = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired session. Please log in again.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
    except jwt.PyJWTError:
        raise invalid
    if not payload.get("sub"):
        raise invalid
    jti = payload.get("jti")
    if jti and is_revoked(jti):
        raise invalid
    return payload


def get_claims(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Return the verified, non-revoked token payload (used by logout)."""
    return _decode(credentials, settings)


def get_current_user(payload: dict = Depends(get_claims)) -> User:
    """Decode the Bearer token and return its user, or raise 401."""
    return User(
        id=payload["sub"],
        username=payload.get("username", ""),
        email=payload.get("email", ""),
    )
