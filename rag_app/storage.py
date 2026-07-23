"""Per-user object storage on MinIO (or any S3-compatible service).

What lives here:

* **Document originals** — the exact bytes a user uploaded. Ingestion only kept
  the derived chunks (the temp file was deleted), so originals were previously
  unrecoverable; now they can be re-downloaded or re-ingested. Qdrant still owns
  the chunks and vectors.
* **Chat snapshots** — a JSON copy of each conversation, written by
  ``rag_app.chat.mirror`` whenever that conversation changes. PostgreSQL remains the
  live read/write store; these are the durable per-user copy.

Key layout — every key starts with the owner's user id, so isolation is
structural (you cannot name another user's object without their id) rather than
a filter each call site has to remember to apply:

    users/{user_id}/chats/{conversation_id}.json            chat snapshot
    users/{user_id}/chats/{conversation_id}/docs/{source}   document original

Optional by default: with ``STORAGE_ENABLED=false`` ``get_storage()`` returns
None and every caller falls back to the previous behaviour, so the app and the
test suite run with no MinIO present.
"""

from __future__ import annotations

import io
import logging
import mimetypes
from functools import lru_cache
from pathlib import Path

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

_ROOT = "users"
_DEFAULT_CONTENT_TYPE = "application/octet-stream"


class StorageError(RuntimeError):
    """Raised when an object-storage operation fails."""


# --- Key layout -------------------------------------------------------------
def _segment(value: str, what: str) -> str:
    """Validate one path segment of a key.

    Ids reach us from auth tokens and request paths, so a segment containing a
    slash or ``..`` could otherwise walk out of the owner's prefix and address
    another user's objects. Both are rejected outright rather than escaped.
    """
    value = (value or "").strip()
    if not value:
        raise StorageError(f"Missing {what}.")
    if "/" in value or "\\" in value or value in (".", "..") or value.startswith(".."):
        raise StorageError(f"Invalid {what}: {value!r}")
    return value


def user_prefix(user_id: str) -> str:
    """Everything owned by one user lives under this prefix."""
    return f"{_ROOT}/{_segment(user_id, 'user id')}/"


def chat_key(user_id: str, conversation_id: str) -> str:
    """Key of a conversation's JSON snapshot."""
    return f"{user_prefix(user_id)}chats/{_segment(conversation_id, 'conversation id')}.json"


def chat_docs_prefix(user_id: str, conversation_id: str) -> str:
    """Prefix holding every document original uploaded into one chat."""
    return f"{user_prefix(user_id)}chats/{_segment(conversation_id, 'conversation id')}/docs/"


def document_key(user_id: str, conversation_id: str, source: str) -> str:
    """Key of one document original.

    ``source`` is the stored document name — the same value Qdrant records and
    the API lists, so a document found in a listing maps to its object with no
    extra bookkeeping. It may contain slashes (a zip member is stored as
    ``archive.zip/inner/file.txt``), which S3 treats as ordinary key text.
    """
    source = (source or "").strip().strip("/")
    if not source or ".." in Path(source).parts:
        raise StorageError(f"Invalid document source: {source!r}")
    return f"{chat_docs_prefix(user_id, conversation_id)}{source}"


def guess_content_type(name: str) -> str:
    return mimetypes.guess_type(name)[0] or _DEFAULT_CONTENT_TYPE


# --- Client -----------------------------------------------------------------
class ObjectStore:
    """Thin wrapper over the MinIO client, holding the bucket and error handling."""

    def __init__(self, settings: Settings | None = None) -> None:
        from minio import Minio  # imported here so the dep is only needed when enabled

        self.settings = settings or get_settings()
        self.bucket = self.settings.minio_bucket
        self.client = Minio(
            self.settings.minio_endpoint,
            access_key=self.settings.minio_access_key,
            secret_key=self.settings.minio_secret_key,
            secure=self.settings.minio_secure,
            region=self.settings.minio_region or None,
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        from minio.error import S3Error

        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info("Created MinIO bucket %r", self.bucket)
        except S3Error as exc:
            raise StorageError(f"Cannot reach bucket {self.bucket!r}: {exc}") from exc

    # --- Writes ---------------------------------------------------------
    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None:
        from minio.error import S3Error

        try:
            self.client.put_object(
                self.bucket,
                key,
                io.BytesIO(data),
                length=len(data),
                content_type=content_type or guess_content_type(key),
            )
        except S3Error as exc:
            raise StorageError(f"Failed to store {key!r}: {exc}") from exc

    def put_file(self, key: str, path: str | Path) -> None:
        """Upload a file from disk (streamed, so upload size is not held in memory)."""
        from minio.error import S3Error

        try:
            self.client.fput_object(
                self.bucket, key, str(path), content_type=guess_content_type(key)
            )
        except S3Error as exc:
            raise StorageError(f"Failed to store {key!r}: {exc}") from exc

    # --- Reads ----------------------------------------------------------
    def get_bytes(self, key: str) -> bytes | None:
        """Return an object's bytes, or None if it does not exist.

        Reading fully into memory is safe here: the only objects are chat
        snapshots and uploads, and uploads are capped at ``MAX_UPLOAD_MB``.
        """
        from minio.error import S3Error

        response = None
        try:
            response = self.client.get_object(self.bucket, key)
            return response.read()
        except S3Error as exc:
            if exc.code in ("NoSuchKey", "NoSuchObject"):
                return None
            raise StorageError(f"Failed to read {key!r}: {exc}") from exc
        finally:
            if response is not None:
                response.close()
                response.release_conn()

    def list_keys(self, prefix: str) -> list[str]:
        from minio.error import S3Error

        try:
            return [
                obj.object_name
                for obj in self.client.list_objects(self.bucket, prefix=prefix, recursive=True)
            ]
        except S3Error as exc:
            raise StorageError(f"Failed to list {prefix!r}: {exc}") from exc

    # --- Deletes --------------------------------------------------------
    def delete(self, key: str) -> None:
        from minio.error import S3Error

        try:
            self.client.remove_object(self.bucket, key)
        except S3Error as exc:
            raise StorageError(f"Failed to delete {key!r}: {exc}") from exc

    def delete_prefix(self, prefix: str) -> int:
        """Delete every object under ``prefix``. Returns how many were removed."""
        from minio.deleteobjects import DeleteObject
        from minio.error import S3Error

        keys = self.list_keys(prefix)
        if not keys:
            return 0
        try:
            errors = list(
                self.client.remove_objects(
                    self.bucket, (DeleteObject(k) for k in keys)
                )
            )
        except S3Error as exc:
            raise StorageError(f"Failed to delete {prefix!r}: {exc}") from exc
        for err in errors:  # remove_objects reports per-object failures lazily
            logger.warning("Could not delete %s: %s", err.name, err.message)
        return len(keys) - len(errors)


@lru_cache
def get_storage() -> ObjectStore | None:
    """Return the process-wide store, or None when storage is disabled.

    A misconfigured or unreachable MinIO returns None (with a warning) rather
    than raising: object storage is a durability layer, so losing it must not
    take chat and retrieval down with it. Callers treat None as "skip".
    """
    settings = get_settings()
    if not settings.storage_enabled:
        return None
    try:
        return ObjectStore(settings)
    except Exception as exc:  # noqa: BLE001 - includes minio import + connection errors
        logger.warning(
            "Object storage is enabled but unavailable (%s); documents and chats "
            "will not be archived to MinIO.", exc,
        )
        return None
