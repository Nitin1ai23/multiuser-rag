"""Qdrant vector store wrapper with per-user isolation.

Runs in embedded on-disk mode (no Docker required). A single shared client is
used process-wide because embedded Qdrant locks its storage directory.

Multi-tenancy: every point carries a ``user_id`` in its payload, and EVERY
read/delete is filtered by ``user_id``. There is no method that returns points
across users, so one user can never retrieve another user's chunks.

Chat-scoped ingestion: a point may also carry a ``conversation_id`` naming the
chat it was uploaded into. When a read/delete passes a ``conversation_id`` the
filter is narrowed to that chat, so documents ingested in one chat are invisible
to another. Passing ``conversation_id=None`` keeps the legacy per-user scope used
by the desktop app (and matches the NULL-conversation convention in the chat
history / message tables).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from .config import Settings, get_settings


@dataclass
class RetrievedChunk:
    text: str
    score: float
    source: str
    metadata: dict


class QdrantStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        # Server mode (qdrant_url set) lifts the single-process lock; otherwise
        # embedded on-disk mode persists to a local directory with no server.
        self.server_mode = bool(self.settings.qdrant_url)
        if self.server_mode:
            self.client = QdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key or None,
            )
        else:
            self.client = QdrantClient(path=self.settings.qdrant_path)
        self.collection = self.settings.qdrant_collection
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.settings.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
        if self.server_mode:
            # Embedded Qdrant ignores payload indexes, but a server needs one
            # on "user_id" so per-tenant filters stay fast as data grows.
            # Idempotent: a no-op if the index already exists.
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name="user_id",
                    field_schema="keyword",
                )
            except Exception:  # noqa: BLE001 - index may already exist
                pass

    @staticmethod
    def _user_filter(
        user_id: str,
        source: str | None = None,
        conversation_id: str | None = None,
    ) -> Filter:
        must = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        if conversation_id is not None:
            must.append(
                FieldCondition(
                    key="conversation_id", match=MatchValue(value=conversation_id)
                )
            )
        if source is not None:
            must.append(FieldCondition(key="source", match=MatchValue(value=source)))
        return Filter(must=must)

    def upsert(
        self,
        texts: list[str],
        vectors: list[list[float]],
        user_id: str,
        source: str,
        conversation_id: str | None = None,
    ) -> int:
        """Store chunks owned by ``user_id``. Returns number of points written.

        When ``conversation_id`` is given the chunks are tagged with it, scoping
        them to a single chat; otherwise they are user-global (desktop legacy).
        """
        points: list[PointStruct] = []
        for text, vector in zip(texts, vectors, strict=True):
            payload = {"text": text, "source": source, "user_id": user_id}
            if conversation_id is not None:
                payload["conversation_id"] = conversation_id
            points.append(
                PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
            )
        if points:
            self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def search(
        self,
        query_vector: list[float],
        user_id: str,
        top_k: int,
        score_threshold: float | None = None,
        conversation_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """Search ONLY within ``user_id``'s chunks (optionally one chat)."""
        hits = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=self._user_filter(user_id, conversation_id=conversation_id),
            limit=top_k,
            with_payload=True,
            score_threshold=score_threshold,
        ).points
        results: list[RetrievedChunk] = []
        for h in hits:
            payload = dict(h.payload or {})
            text = payload.pop("text", "")
            source = payload.pop("source", "unknown")
            payload.pop("user_id", None)
            payload.pop("conversation_id", None)
            results.append(
                RetrievedChunk(text=text, score=h.score, source=source, metadata=payload)
            )
        return results

    def list_sources(
        self, user_id: str, conversation_id: str | None = None
    ) -> list[tuple[str, int]]:
        """Return (source, chunk_count) for each document in scope for ``user_id``.

        Narrowed to one chat when ``conversation_id`` is given.
        """
        counts: dict[str, int] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=self._user_filter(
                    user_id, conversation_id=conversation_id
                ),
                with_payload=["source"],
                limit=256,
                offset=offset,
            )
            for p in points:
                src = (p.payload or {}).get("source", "unknown")
                counts[src] = counts.get(src, 0) + 1
            if offset is None:
                break
        return sorted(counts.items())

    def delete_source(
        self, user_id: str, source: str, conversation_id: str | None = None
    ) -> None:
        """Delete one document (all its chunks) for ``user_id``, one chat if given."""
        self.client.delete(
            collection_name=self.collection,
            points_selector=self._user_filter(
                user_id, source=source, conversation_id=conversation_id
            ),
        )

    def delete_conversation(self, user_id: str, conversation_id: str) -> None:
        """Delete every chunk ingested into one chat (when the chat is deleted)."""
        self.client.delete(
            collection_name=self.collection,
            points_selector=self._user_filter(
                user_id, conversation_id=conversation_id
            ),
        )

    def delete_all_for_user(self, user_id: str) -> None:
        """Delete every chunk owned by ``user_id`` (used when deleting an account)."""
        self.client.delete(
            collection_name=self.collection,
            points_selector=self._user_filter(user_id),
        )


@lru_cache
def get_store() -> QdrantStore:
    """Return the process-wide shared store (embedded Qdrant locks its path)."""
    return QdrantStore()
