"""Document endpoints: upload (ingest), list, delete — all per-user.

Ingestion is chat-scoped: every document belongs to one conversation, so a new
chat starts with no documents and can't see what was uploaded into other chats.
Uploads are size-capped and ingested in the background: the file is streamed to
a temp path (preserving the original filename so it becomes the stored
``source``), a job is created, and ingestion runs after the response returns.
Clients poll ``GET /documents/jobs/{id}`` for progress. Listing and deletion go
through the per-user, per-conversation vector store, so users only ever see and
remove documents from the chat they're in.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from urllib.parse import quote

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import Response

from ...auth.service import User
from ...chat.conversations import ConversationStore
from ...config import get_settings
from ...ingest import SUPPORTED_SUFFIXES
from ...rag import RAGPipeline
from ...storage import StorageError, document_key, get_storage, guess_content_type
from ..ingest_jobs import create_job, get_job, run_ingest
from ..schemas import DeleteResponse, DocumentOut, IngestJobOut
from ..security import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

_SUPPORTED = SUPPORTED_SUFFIXES  # text, code, PDF, Office, images, and .zip
_CHUNK = 1024 * 1024  # read uploads a megabyte at a time


def _require_conversation(user_id: str, conversation_id: str | None):
    """Return an owned conversation, 404 for an unknown id, or create a fresh one.

    Uploading in a brand-new (not-yet-saved) chat sends no id, so we create the
    conversation here — that's what binds the document to this chat.
    """
    store = ConversationStore(user_id)
    if conversation_id:
        conv = store.get(conversation_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        return conv
    return store.create()


@router.get("", response_model=list[DocumentOut])
def list_documents(
    conversation_id: str | None = None, user: User = Depends(get_current_user)
) -> list[DocumentOut]:
    # No chat selected (a brand-new chat) means no documents yet.
    if not conversation_id:
        return []
    sources = RAGPipeline(
        user.id, conversation_id=conversation_id
    ).list_documents()
    return [DocumentOut(source=src, chunks=count) for src, count in sources]


@router.post("", response_model=IngestJobOut, status_code=202)
def upload(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    conversation_id: str | None = Form(default=None),
    user: User = Depends(get_current_user),
) -> IngestJobOut:
    conv = _require_conversation(user.id, conversation_id)
    name = Path(file.filename or "upload").name
    suffix = Path(name).suffix.lower()
    if suffix not in _SUPPORTED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{suffix or '(none)'}'. Supported: "
            "text & code files, CSV/JSON/YAML, PDF, Word, Excel, PowerPoint, "
            "images, and .zip archives.",
        )

    max_bytes = get_settings().max_upload_mb * 1024 * 1024

    # Stream to a temp dir under the real filename, enforcing the size cap so a
    # huge upload can't exhaust memory or disk.
    tmp_dir = tempfile.mkdtemp(prefix="rag_upload_")
    tmp_path = Path(tmp_dir) / name
    written = 0
    try:
        with tmp_path.open("wb") as out:
            while chunk := file.file.read(_CHUNK):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds the {get_settings().max_upload_mb} MB limit.",
                    )
                out.write(chunk)
    except HTTPException:
        tmp_path.unlink(missing_ok=True)
        Path(tmp_dir).rmdir()
        raise

    if written == 0:
        tmp_path.unlink(missing_ok=True)
        Path(tmp_dir).rmdir()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file."
        )

    job = create_job(user.id, name, conversation_id=conv.id)
    background.add_task(run_ingest, job.id, str(tmp_path))
    return IngestJobOut(
        job_id=job.id, source=name, status=job.status, conversation_id=conv.id
    )


@router.get("/jobs/{job_id}", response_model=IngestJobOut)
def ingest_status(
    job_id: str, user: User = Depends(get_current_user)
) -> IngestJobOut:
    job = get_job(job_id, user.id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return IngestJobOut(
        job_id=job.id,
        source=job.source,
        status=job.status,
        conversation_id=job.conversation_id or "",
        chunks_added=job.chunks_added,
        detail=job.detail,
    )


@router.get("/download")
def download_document(
    source: str,
    conversation_id: str,
    user: User = Depends(get_current_user),
) -> Response:
    """Return the original bytes of a document the user uploaded into this chat.

    ``source`` and ``conversation_id`` are query params rather than path
    segments because a source can contain slashes (a zip member is stored as
    ``archive.zip/inner/file.txt``). The key is built from the caller's own user
    id, so it can only ever address their own object.

    Only what was actually uploaded is stored: asking for an individual member
    of a zip 404s, since the archive was stored whole under its own name.
    """
    storage = get_storage()
    if storage is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document storage is not enabled on this server.",
        )
    if ConversationStore(user.id).get(conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    try:
        key = document_key(user.id, conversation_id, source)
    except StorageError as exc:  # a source that tries to escape the user's prefix
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    data = storage.get_bytes(key)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail="No stored original for that document. Files ingested before "
            "storage was enabled, and members of an uploaded archive, have none.",
        )
    name = Path(source).name
    return Response(
        content=data,
        media_type=guess_content_type(name),
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"
        },
    )


@router.delete("/{source}", response_model=DeleteResponse)
def delete_document(
    source: str,
    conversation_id: str,
    user: User = Depends(get_current_user),
) -> DeleteResponse:
    RAGPipeline(user.id, conversation_id=conversation_id).delete_document(source)
    # Drop the archived original too, so a deleted document leaves nothing behind.
    storage = get_storage()
    if storage is not None:
        try:
            storage.delete(document_key(user.id, conversation_id, source))
        except Exception as exc:  # noqa: BLE001 - chunks are gone; that's the delete
            logger.warning("Could not delete stored original %s: %s", source, exc)
    return DeleteResponse(source=source)
