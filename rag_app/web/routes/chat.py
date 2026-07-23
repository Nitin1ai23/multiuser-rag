"""Chat endpoints: conversations, RAG query, and per-conversation history.

The pipeline, history, and conversation store are all constructed with
``current_user.id``, so a request can only ever touch the caller's own vectors,
conversations, and messages. The query endpoint is a sync ``def`` on purpose:
the RAG query makes blocking network calls (embeddings + Groq), and FastAPI runs
sync handlers in a threadpool so they don't block the event loop.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from ...auth.service import User
from ...chat.conversations import ConversationStore, make_title, DEFAULT_TITLE
from ...chat.history import ChatHistory
from ...rag import RAGPipeline
from ...vectorstore import get_store
from ..schemas import (
    ConversationOut,
    CreateConversationRequest,
    MessageOut,
    QueryRequest,
    QueryResponse,
    SourceOut,
)
from ..security import get_current_user

router = APIRouter(prefix="/chat", tags=["chat"])


def _resolve_conversation(store: ConversationStore, conversation_id: str | None):
    """Return an owned conversation, raising 404 for an unknown id, or create one."""
    conv = store.get(conversation_id) if conversation_id else None
    if conversation_id and conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conv or store.create()


def _maybe_title(store: ConversationStore, conv, question: str) -> None:
    if conv.title == DEFAULT_TITLE:
        new_title = make_title(question)
        store.rename(conv.id, new_title)
        conv.title = new_title


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _conv_out(conv) -> ConversationOut:
    return ConversationOut(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
    )


# --- Conversations ------------------------------------------------------
@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(user: User = Depends(get_current_user)) -> list[ConversationOut]:
    return [_conv_out(c) for c in ConversationStore(user.id).list()]


@router.post("/conversations", response_model=ConversationOut, status_code=201)
def create_conversation(
    body: CreateConversationRequest | None = None,
    user: User = Depends(get_current_user),
) -> ConversationOut:
    title = body.title if body else None
    return _conv_out(ConversationStore(user.id).create(title))


@router.get("/conversations/{conversation_id}/messages", response_model=list[MessageOut])
def conversation_messages(
    conversation_id: str, user: User = Depends(get_current_user)
) -> list[MessageOut]:
    store = ConversationStore(user.id)
    if store.get(conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    msgs = ChatHistory(user.id, conversation_id).all()
    return [
        MessageOut(role=m.role, content=m.content, created_at=m.created_at)
        for m in msgs
    ]


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: str, user: User = Depends(get_current_user)
) -> None:
    if not ConversationStore(user.id).delete(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    # Drop the documents ingested into this chat so they don't linger in the store.
    get_store().delete_conversation(user.id, conversation_id)


# --- Query --------------------------------------------------------------
@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, user: User = Depends(get_current_user)) -> QueryResponse:
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Question is empty."
        )

    store = ConversationStore(user.id)
    conv = _resolve_conversation(store, body.conversation_id)

    history = ChatHistory(user.id, conv.id)
    prior = history.all()  # turns before this question, for follow-up context
    history.add("user", question)
    _maybe_title(store, conv, question)

    try:
        result = RAGPipeline(user.id, conversation_id=conv.id).query(
            question, history=prior
        )
    except Exception as err:  # surface a clean message instead of a 500 traceback
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Generation failed: {err}",
        )
    history.add("assistant", result.answer)

    return QueryResponse(
        answer=result.answer,
        provider=result.provider,
        conversation_id=conv.id,
        title=conv.title,
        sources=[
            SourceOut(text=c.text, score=c.score, source=c.source)
            for c in result.sources
        ],
    )


@router.post("/query/stream")
def query_stream(
    body: QueryRequest, user: User = Depends(get_current_user)
) -> StreamingResponse:
    """Stream the answer as Server-Sent Events.

    Event sequence: ``meta`` (conversation id, title, sources) → many ``token``
    events → ``done``; an ``error`` event is emitted instead if generation fails.
    """
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Question is empty."
        )

    store = ConversationStore(user.id)
    conv = _resolve_conversation(store, body.conversation_id)
    history = ChatHistory(user.id, conv.id)
    prior = history.all()
    history.add("user", question)
    _maybe_title(store, conv, question)

    def event_stream():
        try:
            chunks, tokens = RAGPipeline(
                user.id, conversation_id=conv.id
            ).query_stream(question, history=prior)
        except Exception as err:  # noqa: BLE001
            yield _sse("error", {"detail": f"Generation failed: {err}"})
            return

        yield _sse("meta", {
            "conversation_id": conv.id,
            "title": conv.title,
            "sources": [
                {"text": c.text, "score": c.score, "source": c.source}
                for c in chunks
            ],
        })

        parts: list[str] = []
        try:
            for tok in tokens:
                parts.append(tok)
                yield _sse("token", {"text": tok})
        except Exception as err:  # noqa: BLE001
            yield _sse("error", {"detail": f"Generation failed: {err}"})
        finally:
            if parts:  # persist whatever we managed to generate
                history.add("assistant", "".join(parts))
        yield _sse("done", {})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
