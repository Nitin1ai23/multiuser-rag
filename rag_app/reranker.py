"""Cross-encoder reranking via NVIDIA NIM.

Dense vector search is recall-oriented: it returns the *k* nearest chunks, but a
nearest chunk isn't always the most relevant one. A reranker re-scores each
(query, passage) pair with a cross-encoder, which is far more precise than the
cosine similarity of two independent embeddings.

The flow (see ``rag.RAGPipeline``) is: retrieve ``top_k * multiplier``
candidates by vector, rerank them here, then keep the best ``top_k``.

This is best-effort: if reranking is disabled, unconfigured, or the API call
fails, callers fall back to the original vector order so a query still answers.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import httpx

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


class NIMReranker:
    """Wraps the NVIDIA NIM ``/ranking`` endpoint."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model = self.settings.nim_rerank_model
        self.url = (
            self.settings.nim_rerank_url
            or self.settings.nim_base_url.rstrip("/") + "/ranking"
        )

    def rerank(self, query: str, passages: list[str]) -> list[tuple[int, float]]:
        """Return ``(original_index, score)`` pairs, best first.

        On any failure returns the passages in their original order with a
        score of 0.0 so retrieval still succeeds.
        """
        if not passages:
            return []
        fallback = [(i, 0.0) for i in range(len(passages))]
        if not self.settings.nvidia_api_key:
            return fallback
        try:
            resp = httpx.post(
                self.url,
                headers={
                    "Authorization": f"Bearer {self.settings.nvidia_api_key}",
                    "Accept": "application/json",
                },
                json={
                    "model": self.model,
                    "query": {"text": query},
                    "passages": [{"text": p} for p in passages],
                    "truncate": "END",
                },
                timeout=20.0,
            )
            resp.raise_for_status()
            rankings = resp.json().get("rankings", [])
            if not rankings:
                return fallback
            return [
                (int(r["index"]), float(r.get("logit", 0.0)))
                for r in rankings
                if 0 <= int(r["index"]) < len(passages)
            ]
        except Exception as exc:  # noqa: BLE001 - reranking is best-effort
            logger.warning("Reranking failed, using vector order: %s", exc)
            return fallback


@lru_cache
def get_reranker() -> NIMReranker:
    """Return the process-wide cached reranker."""
    return NIMReranker()
