"""Auth endpoints: signup, login, security-question password reset.

Thin wrappers over :class:`rag_app.auth.service.AuthService`. ``AuthError`` is
translated into a 400 with its user-facing message.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from ...auth.service import SECURITY_QUESTIONS, AuthError, AuthService, User
from ...storage import get_storage, user_prefix
from ...vectorstore import get_store
from ..ratelimit import auth_rate_limit
from ..schemas import (
    DeleteAccountRequest,
    ForgotQuestionRequest,
    LoginRequest,
    MessageResponse,
    ResetPasswordRequest,
    SecurityQuestionResponse,
    SecurityQuestionsResponse,
    SignupRequest,
    TokenResponse,
    UserOut,
)
from ..security import (
    create_access_token,
    get_claims,
    get_current_user,
    revoke_jti,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


def get_auth() -> AuthService:
    return AuthService()


def _token(user: User) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(user),
        user=UserOut(id=user.id, username=user.username, email=user.email),
    )


def _bad_request(err: AuthError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))


@router.get("/security-questions", response_model=SecurityQuestionsResponse)
def security_questions() -> SecurityQuestionsResponse:
    """The preset security questions to pick from at signup."""
    return SecurityQuestionsResponse(security_questions=list(SECURITY_QUESTIONS))


@router.post(
    "/signup",
    response_model=TokenResponse,
    status_code=201,
    dependencies=[Depends(auth_rate_limit)],
)
def signup(body: SignupRequest, auth: AuthService = Depends(get_auth)) -> TokenResponse:
    try:
        user = auth.signup(
            username=body.username,
            email=body.email,
            password=body.password,
            security_question=body.security_question,
            security_answer=body.security_answer,
        )
    except AuthError as err:
        raise _bad_request(err)
    return _token(user)


@router.post(
    "/login",
    response_model=TokenResponse,
    dependencies=[Depends(auth_rate_limit)],
)
def login(body: LoginRequest, auth: AuthService = Depends(get_auth)) -> TokenResponse:
    try:
        user = auth.login(body.identifier, body.password)
    except AuthError as err:
        # Credentials failures are 401, not 400.
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(err))
    return _token(user)


@router.post(
    "/forgot",
    response_model=SecurityQuestionResponse,
    dependencies=[Depends(auth_rate_limit)],
)
def forgot(
    body: ForgotQuestionRequest, auth: AuthService = Depends(get_auth)
) -> SecurityQuestionResponse:
    try:
        question = auth.get_security_question(body.identifier)
    except AuthError as err:
        raise _bad_request(err)
    return SecurityQuestionResponse(security_question=question)


@router.post(
    "/reset",
    response_model=TokenResponse,
    dependencies=[Depends(auth_rate_limit)],
)
def reset(
    body: ResetPasswordRequest, auth: AuthService = Depends(get_auth)
) -> TokenResponse:
    try:
        user = auth.reset_password(
            body.identifier, body.security_answer, body.new_password
        )
    except AuthError as err:
        raise _bad_request(err)
    return _token(user)


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)) -> UserOut:
    """Validate the current token and echo back the user (used on app load)."""
    return UserOut(id=user.id, username=user.username, email=user.email)


@router.post("/logout", response_model=MessageResponse)
def logout(claims: dict = Depends(get_claims)) -> MessageResponse:
    """Revoke the current token so it can no longer be used."""
    jti, exp = claims.get("jti"), claims.get("exp")
    if jti and exp:
        revoke_jti(jti, int(exp))
    return MessageResponse(detail="Signed out.")


@router.delete("/me", response_model=MessageResponse)
def delete_account(
    body: DeleteAccountRequest,
    claims: dict = Depends(get_claims),
    auth: AuthService = Depends(get_auth),
) -> MessageResponse:
    """Permanently delete the account, its documents, chats, and vectors."""
    user_id = claims["sub"]
    try:
        auth.delete_account(user_id, body.password)
    except AuthError as err:
        raise _bad_request(err)
    # Remove the user's vectors and stored objects, and revoke the token now
    # that the account is gone.
    get_store().delete_all_for_user(user_id)
    storage = get_storage()
    if storage is not None:
        try:
            storage.delete_prefix(user_prefix(user_id))
        except Exception as exc:  # noqa: BLE001 - the account is already deleted
            logger.warning("Could not purge stored objects for %s: %s", user_id, exc)
    jti, exp = claims.get("jti"), claims.get("exp")
    if jti and exp:
        revoke_jti(jti, int(exp))
    return MessageResponse(detail="Account deleted.")
