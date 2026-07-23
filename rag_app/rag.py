"""RAG orchestrator, scoped to a single user.

A ``RAGPipeline`` is bound to one ``user_id`` at construction. Every ingest and
query call passes that id to the vector store, so the pipeline physically cannot
read or write another user's data.

Pipeline:
    ingest:  load -> chunk -> NIM embed (passage) -> Qdrant upsert (user_id)
             images take a detour first: Groq vision caption + OCR, so they are
             indexed by what they show rather than by their characters alone
    query:   (condense w/ history) -> NIM embed (query) -> Qdrant search (user_id)
             -> NIM rerank -> prompt (w/ history) -> Groq

Answers are grounded in the user's documents, but a question those documents
don't cover does not dead-end: the model falls back to a brief general-knowledge
answer, flags that it isn't from their documents, and invites an upload (see
SYSTEM_PROMPT). Retrieval returning nothing and retrieval returning irrelevant
chunks both land in that branch — the first is stated in the prompt, the second
is a judgement only the model can make.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings, get_settings
from .embeddings import get_embedder
from .ingest import (
    IMAGE_SUFFIXES,
    UnsupportedFileError,
    chunk_text,
    load_documents,
    load_text,
)
from .llm import get_provider
from .reranker import get_reranker
from .vectorstore import RetrievedChunk, get_store

logger = logging.getLogger(__name__)

# A turn of prior conversation, as (role, content). Accepts ChatHistory.Message
# too (it exposes .role/.content) — see _normalize_history.
Turn = tuple[str, str]

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about the user's uploaded "
    "documents.\n\n"
    "If the context below answers the question, answer from that context alone "
    "and cite the source filename in brackets, e.g. [report.pdf].\n\n"
    "If the context does not answer the question, or no context was retrieved, "
    "do NOT refuse and do NOT stop at 'I don't know'. Instead:\n"
    "1. In one short line, tell the user their documents don't cover this. "
    "Address them directly as 'you' — never talk about 'the user'.\n"
    "2. Answer the question anyway from your general knowledge — a few sentences "
    "that give a genuinely useful interpretation of what they asked. Say once "
    "that this part is general knowledge rather than from their documents.\n"
    "3. Close by inviting them to upload a relevant document for a detailed, "
    "cited answer.\n\n"
    "Never cite a filename for anything that did not come from the context, and "
    "never present general knowledge as though it came from their documents. If "
    "the message is a greeting or small talk, just reply naturally and skip all "
    "of the above."
)

# Shown to the model in place of a context block when retrieval came back empty:
# the user has uploaded nothing to this chat, or nothing cleared the score
# threshold. Phrased for both cases, and states the absence plainly so the model
# takes the fallback branch of SYSTEM_PROMPT rather than inventing a source.
_NO_CONTEXT = (
    "No context was retrieved from the user's documents for this question — "
    "they have not uploaded anything relevant to this chat."
)

_CONDENSE_SYSTEM = (
    "Given a conversation and a follow-up question, rewrite the follow-up as a "
    "standalone question that captures any context it relies on from the "
    "conversation. Reply with ONLY the rewritten question and nothing else. If "
    "it is already standalone, return it unchanged."
)


@dataclass
class Answer:
    answer: str
    provider: str
    sources: list[RetrievedChunk] = field(default_factory=list)


def _normalize_history(history) -> list[Turn]:
    """Accept tuples or objects with .role/.content; return (role, content)."""
    if not history:
        return []
    out: list[Turn] = []
    for item in history:
        if isinstance(item, (tuple, list)):
            out.append((item[0], item[1]))
        else:  # ChatHistory.Message-like
            out.append((item.role, item.content))
    return out


class RAGPipeline:
    def __init__(
        self,
        user_id: str,
        settings: Settings | None = None,
        conversation_id: str | None = None,
    ) -> None:
        if not user_id:
            raise ValueError("RAGPipeline requires a user_id.")
        self.user_id = user_id
        # Chat-scoped ingestion: when set, ingest tags chunks with this chat and
        # retrieval only sees chunks uploaded into it. None = user-global
        # (desktop legacy), so prior desktop behaviour is unchanged.
        self.conversation_id = conversation_id
        self.settings = settings or get_settings()
        self.embedder = get_embedder()
        self.store = get_store()
        self.reranker = get_reranker()

    # --- Ingestion -------------------------------------------------------
    def ingest_text(self, text: str, source: str) -> int:
        """Chunk, embed, and store a raw text blob for this user.

        Re-ingesting the same ``source`` replaces it (its old chunks are
        removed first) so uploading a document twice never duplicates it.
        """
        chunks = chunk_text(text, self.settings)
        if not chunks:
            return 0
        self.store.delete_source(
            self.user_id, source, conversation_id=self.conversation_id
        )
        vectors = self.embedder.embed_documents(chunks)
        return self.store.upsert(
            chunks, vectors, user_id=self.user_id, source=source,
            conversation_id=self.conversation_id,
        )

    def _ingest_image(self, path: Path) -> int:
        """Index an image by what it *shows*, not just the characters in it.

        OCR alone leaves a photo or chart with no indexable text at all, so the
        only possible answer about it is "I don't know". A vision caption fixes
        that; OCR still runs because it reads small print (axis labels, serial
        numbers) more reliably than a captioner does.
        """
        data = path.read_bytes()
        blocks: list[str] = []

        if self.settings.vision_enabled:
            try:
                caption = get_provider(self.settings).describe_image(data)
                if caption.strip():
                    blocks.append(f"Image description: {caption.strip()}")
            except Exception as exc:  # noqa: BLE001 - OCR alone is still useful
                logger.warning("Vision captioning failed for %s: %s", path.name, exc)

        try:
            ocr = load_text(path).strip()
        except UnsupportedFileError:
            if not blocks:
                raise  # unreadable by both paths — let the caller report it
            ocr = ""
        if ocr:
            blocks.append(f"Text in image: {ocr}")

        if not blocks:
            return 0
        return self.ingest_text("\n\n".join(blocks), path.name)

    def ingest_file(self, path: str | Path) -> int:
        """Load, chunk, embed, and store a file for this user.

        Most files become a single document; an archive (.zip) expands into one
        document per readable member, each stored under its own ``source`` so
        citations stay precise. Returns the total chunks written across all
        documents. Re-ingesting a source replaces its previous version.
        """
        path = Path(path)
        if path.suffix.lower() in IMAGE_SUFFIXES:
            return self._ingest_image(path)
        total = 0
        for source, text in load_documents(path):
            total += self.ingest_text(text, source)
        return total

    # --- Documents -------------------------------------------------------
    def list_documents(self) -> list[tuple[str, int]]:
        return self.store.list_sources(
            self.user_id, conversation_id=self.conversation_id
        )

    def delete_document(self, source: str) -> None:
        self.store.delete_source(
            self.user_id, source, conversation_id=self.conversation_id
        )

    # --- Querying --------------------------------------------------------
    def _recent_history(self, history) -> list[Turn]:
        turns = _normalize_history(history)
        n = self.settings.history_turns
        return turns[-n:] if n > 0 else []

    def _condense(self, question: str, turns: list[Turn]) -> str:
        """Rewrite a follow-up into a standalone question using prior turns."""
        if not turns:
            return question
        convo = "\n".join(f"{role}: {content}" for role, content in turns)
        prompt = f"Conversation:\n{convo}\n\nFollow-up question: {question}"
        try:
            rewritten = get_provider(self.settings).generate(_CONDENSE_SYSTEM, prompt)
            return rewritten.strip() or question
        except Exception as exc:  # noqa: BLE001 - fall back to raw question
            logger.warning("Condense failed, using raw question: %s", exc)
            return question

    def retrieve(
        self, question: str, turns: list[Turn] | None = None
    ) -> list[RetrievedChunk]:
        """Embed, vector-search, then rerank down to ``top_k`` for this user."""
        top_k = self.settings.top_k
        standalone = self._condense(question, turns or [])
        query_vec = self.embedder.embed_query(standalone)

        threshold = self.settings.retrieval_score_threshold or None
        if self.settings.rerank_enabled:
            candidate_k = max(top_k, top_k * self.settings.rerank_candidate_multiplier)
            candidates = self.store.search(
                query_vec, user_id=self.user_id, top_k=candidate_k,
                conversation_id=self.conversation_id,
            )
            ranked = self.reranker.rerank(standalone, [c.text for c in candidates])
            reordered: list[RetrievedChunk] = []
            for idx, score in ranked:
                chunk = candidates[idx]
                chunk.score = score
                reordered.append(chunk)
            chunks = reordered[:top_k]
            if threshold is not None:
                chunks = [c for c in chunks if c.score >= threshold]
            return chunks

        return self.store.search(
            query_vec, user_id=self.user_id, top_k=top_k, score_threshold=threshold,
            conversation_id=self.conversation_id,
        )

    def _build_prompt(
        self, question: str, chunks: list[RetrievedChunk], turns: list[Turn]
    ) -> str:
        if chunks:
            blocks = [f"[Source: {c.source}]\n{c.text}" for c in chunks]
            parts = ["Context:\n" + "\n\n---\n\n".join(blocks)]
        else:
            parts = [_NO_CONTEXT]
        if turns:
            convo = "\n".join(f"{role}: {content}" for role, content in turns)
            parts.append(f"Conversation so far:\n{convo}")
        parts.append(f"Question: {question}\n\nAnswer:")
        return "\n\n".join(parts)

    def query(
        self,
        question: str,
        top_k: int | None = None,  # kept for API compatibility; settings.top_k is used
        history: Sequence | None = None,
    ) -> Answer:
        """Answer a question using the user's documents and recent history."""
        started = time.perf_counter()
        turns = self._recent_history(history)
        chunks = self.retrieve(question, turns)
        prompt = self._build_prompt(question, chunks, turns)
        llm = get_provider(self.settings)
        text = llm.generate(SYSTEM_PROMPT, prompt)
        self._log_query(question, chunks, started)
        return Answer(answer=text, provider=llm.name, sources=chunks)

    def query_stream(
        self, question: str, history: Sequence | None = None
    ) -> tuple[list[RetrievedChunk], Iterator[str]]:
        """Retrieve sources, then return them with a token iterator.

        Sources are resolved up front so the caller can emit them before the
        answer streams. Iterating the returned generator drives generation.
        """
        started = time.perf_counter()
        turns = self._recent_history(history)
        chunks = self.retrieve(question, turns)
        prompt = self._build_prompt(question, chunks, turns)
        llm = get_provider(self.settings)

        def tokens() -> Iterator[str]:
            yield from llm.generate_stream(SYSTEM_PROMPT, prompt)
            self._log_query(question, chunks, started)

        return chunks, tokens()

    def _log_query(self, question: str, chunks, started: float) -> None:
        if not self.settings.log_queries:
            return
        logger.info(
            "rag.query user=%s chunks=%d elapsed_ms=%d q=%r",
            self.user_id, len(chunks),
            int((time.perf_counter() - started) * 1000),
            question[:120],
        )
