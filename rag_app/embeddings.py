"""Embedding generation via NVIDIA NIM (OpenAI-compatible endpoint).

Groq has no embeddings API, so retrieval embeddings come from NIM. The
nv-embedqa family requires an ``input_type`` of either ``"query"`` or
``"passage"`` so that asymmetric query/document embeddings are produced.
"""

from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from .config import Settings, get_settings


class NIMEmbedder:
    """Wraps the NVIDIA NIM embeddings endpoint."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.nvidia_api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY is not set. Add it to your .env file "
                "(get one at https://build.nvidia.com)."
            )
        self.client = OpenAI(
            base_url=self.settings.nim_base_url,
            api_key=self.settings.nvidia_api_key,
        )
        self.model = self.settings.nim_embedding_model

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        kwargs: dict = {"model": self.model, "input": texts}
        if self.settings.nim_embedding_requires_input_type:
            # NIM accepts input_type via extra_body for the OpenAI client.
            kwargs["extra_body"] = {"input_type": input_type, "truncate": "END"}
        resp = self.client.embeddings.create(**kwargs)
        # Preserve request order.
        return [item.embedding for item in sorted(resp.data, key=lambda d: d.index)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages for storage. Batches to respect API limits."""
        if not texts:
            return []
        out: list[list[float]] = []
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            out.extend(self._embed(texts[i : i + batch_size], input_type="passage"))
        return out

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self._embed([text], input_type="query")[0]


@lru_cache
def get_embedder() -> NIMEmbedder:
    """Return a process-wide cached embedder (one OpenAI client)."""
    return NIMEmbedder()
