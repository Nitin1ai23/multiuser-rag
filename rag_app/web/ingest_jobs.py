"""In-memory registry for background document-ingestion jobs.

Ingesting a large PDF means chunking + many embedding calls, which can take long
enough to time out a request. So uploads are accepted, persisted to a temp file,
and processed in a background task; the client polls a job id for progress.

State lives in memory: fine for the single-worker deployment this app targets
(embedded Qdrant locks its directory). Completed/failed jobs are pruned after a
short TTL so the dict doesn't grow unbounded.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from ..rag import RAGPipeline
from ..storage import document_key, get_storage

logger = logging.getLogger(__name__)

_JOB_TTL = 3600.0  # keep finished jobs queryable for an hour


@dataclass
class IngestJob:
    id: str
    user_id: str
    source: str
    conversation_id: str | None = None  # chat this document was uploaded into
    status: str = "pending"  # pending | running | done | error
    chunks_added: int = 0
    detail: str = ""
    updated_at: float = field(default_factory=time.time)


_lock = threading.Lock()
_jobs: dict[str, IngestJob] = {}


def _prune(now: float) -> None:
    stale = [
        jid for jid, j in _jobs.items()
        if j.status in ("done", "error") and now - j.updated_at > _JOB_TTL
    ]
    for jid in stale:
        _jobs.pop(jid, None)


def create_job(
    user_id: str, source: str, conversation_id: str | None = None
) -> IngestJob:
    job = IngestJob(
        id=uuid.uuid4().hex,
        user_id=user_id,
        source=source,
        conversation_id=conversation_id,
    )
    with _lock:
        _prune(time.time())
        _jobs[job.id] = job
    return job


def get_job(job_id: str, user_id: str) -> IngestJob | None:
    """Return a job only if it belongs to ``user_id`` (per-user isolation)."""
    with _lock:
        job = _jobs.get(job_id)
    if job is None or job.user_id != user_id:
        return None
    return job


def _update(job: IngestJob, **changes) -> None:
    with _lock:
        for k, v in changes.items():
            setattr(job, k, v)
        job.updated_at = time.time()


def _archive_original(job: IngestJob, path: Path) -> str:
    """Store the uploaded file's original bytes under the owner's prefix.

    Returns a note for the job detail if archiving didn't happen. Ingestion
    continues either way: a MinIO outage should cost the user the ability to
    re-download the original later, not the ability to ask questions about it.
    """
    storage = get_storage()
    if storage is None or not job.conversation_id:
        return ""
    try:
        storage.put_file(
            document_key(job.user_id, job.conversation_id, job.source), path
        )
        return ""
    except Exception as exc:  # noqa: BLE001 - archive is best-effort
        logger.warning("Could not archive %s for user %s: %s", job.source, job.user_id, exc)
        return "Indexed, but the original file could not be saved to storage."


def run_ingest(job_id: str, tmp_path: str) -> None:
    """Background entry point: archive + ingest ``tmp_path``, then delete it."""
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        return
    _update(job, status="running")
    path = Path(tmp_path)
    try:
        note = _archive_original(job, path)
        added = RAGPipeline(
            job.user_id, conversation_id=job.conversation_id
        ).ingest_file(path)
        if added == 0:
            _update(
                job, status="error",
                detail="No text could be extracted from that file.",
            )
        else:
            _update(job, status="done", chunks_added=added, detail=note)
    except Exception as exc:  # noqa: BLE001 - surface to the client as job state
        logger.exception("Ingestion job %s failed", job_id)
        _update(job, status="error", detail=f"Ingestion failed: {exc}")
    finally:
        try:
            path.unlink(missing_ok=True)
            path.parent.rmdir()
        except OSError:
            pass
