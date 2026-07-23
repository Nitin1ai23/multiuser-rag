import pytest

from rag_app.auth.service import SECURITY_QUESTIONS, AuthError, AuthService

_QUESTION = SECURITY_QUESTIONS[0]


def _signup(svc, username="alice", email=None):
    return svc.signup(
        username=username,
        email=email or f"{username}@example.com",
        password="password123",
        security_question=_QUESTION,
        security_answer="Rex",
    )


def test_signup_then_login_by_username_or_email():
    svc = AuthService()
    user = _signup(svc)
    assert svc.login("alice", "password123").id == user.id
    assert svc.login("alice@example.com", "password123").id == user.id


def test_login_rejects_bad_password_and_unknown_user():
    svc = AuthService()
    _signup(svc)
    with pytest.raises(AuthError):
        svc.login("alice", "wrong-password")
    with pytest.raises(AuthError):
        svc.login("nobody", "password123")


def test_duplicate_username_is_rejected():
    svc = AuthService()
    _signup(svc)
    with pytest.raises(AuthError):
        _signup(svc, username="alice", email="other@example.com")


def test_signup_rejects_question_not_in_preset_list():
    svc = AuthService()
    with pytest.raises(AuthError):
        svc.signup(
            username="mallory",
            email="mallory@example.com",
            password="password123",
            security_question="My own custom question?",
            security_answer="Rex",
        )


def test_password_reset_via_security_answer_is_case_insensitive():
    svc = AuthService()
    user = _signup(svc)
    assert svc.get_security_question("alice") == _QUESTION
    svc.reset_password("alice", "rex", "newpassword1")  # lower-case answer
    assert svc.login("alice", "newpassword1").id == user.id
    with pytest.raises(AuthError):
        svc.login("alice", "password123")  # old password no longer works


def test_delete_account_requires_correct_password():
    svc = AuthService()
    user = _signup(svc, username="bob")
    with pytest.raises(AuthError):
        svc.delete_account(user.id, "wrong-password")
    svc.delete_account(user.id, "password123")
    with pytest.raises(AuthError):
        svc.login("bob", "password123")
