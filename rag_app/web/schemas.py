"""Pydantic request/response models for the web API."""

from __future__ import annotations

from pydantic import BaseModel, Field


# --- Auth ---------------------------------------------------------------
class SignupRequest(BaseModel):
    username: str
    email: str
    password: str
    security_question: str
    security_answer: str


class LoginRequest(BaseModel):
    identifier: str = Field(..., description="username or email")
    password: str


class ForgotQuestionRequest(BaseModel):
    identifier: str


class ResetPasswordRequest(BaseModel):
    identifier: str
    security_answer: str
    new_password: str


class UserOut(BaseModel):
    id: str
    username: str
    email: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class SecurityQuestionResponse(BaseModel):
    security_question: str


class SecurityQuestionsResponse(BaseModel):
    security_questions: list[str]


class DeleteAccountRequest(BaseModel):
    password: str


class MessageResponse(BaseModel):
    detail: str


# --- Chat ---------------------------------------------------------------
class ConversationOut(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class CreateConversationRequest(BaseModel):
    title: str | None = None


class QueryRequest(BaseModel):
    question: str
    conversation_id: str | None = None
    top_k: int | None = None


class SourceOut(BaseModel):
    text: str
    score: float
    source: str


class QueryResponse(BaseModel):
    answer: str
    provider: str
    sources: list[SourceOut]
    conversation_id: str
    title: str


class MessageOut(BaseModel):
    role: str
    content: str
    created_at: str


# --- Documents ----------------------------------------------------------
class DocumentOut(BaseModel):
    source: str
    chunks: int


class IngestResponse(BaseModel):
    source: str
    chunks_added: int


class IngestJobOut(BaseModel):
    job_id: str
    source: str
    status: str            # pending | running | done | error
    conversation_id: str   # the chat this document was ingested into
    chunks_added: int = 0
    detail: str = ""


class DeleteResponse(BaseModel):
    source: str
    deleted: bool = True
