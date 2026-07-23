"""Central configuration loaded from environment / .env file.

All tunables live here so the rest of the code never reads os.environ directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API credentials -------------------------------------------------
    # NVIDIA NIM -> embeddings only.  Groq -> chat generation.
    nvidia_api_key: str = ""
    groq_api_key: str = ""

    # --- NVIDIA NIM embeddings ------------------------------------------
    nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    nim_embedding_model: str = "nvidia/llama-nemotron-embed-1b-v2"
    # nv-embedqa models require an input_type ("query" vs "passage").
    nim_embedding_requires_input_type: bool = True
    embedding_dim: int = 2048

    # --- NVIDIA NIM reranking (optional, improves retrieval precision) ---
    # After dense retrieval we re-score candidates with a cross-encoder and
    # keep the best `top_k`. Always falls back to vector order if the call
    # fails, so it's safe to leave on. Off by default because NVIDIA's hosted
    # rerank endpoint is mid-migration (the long-standing URL was retired
    # 2026-05-18); set NIM_RERANK_URL to a working endpoint (hosted or a
    # self-hosted NIM, e.g. http://localhost:8000/v1/ranking) and flip this on.
    rerank_enabled: bool = False
    nim_rerank_model: str = "nvidia/llama-3.2-nv-rerankqa-1b-v2"
    # Full reranking endpoint URL. Empty -> derive "{nim_base_url}/ranking".
    nim_rerank_url: str = ""
    # How many candidates to pull before reranking down to top_k.
    rerank_candidate_multiplier: int = 4

    # --- Groq chat -------------------------------------------------------
    groq_model: str = "llama-3.3-70b-versatile"

    # --- Vision (image captioning at ingest) -----------------------------
    # OCR only recovers characters, so a photo or chart carrying no text
    # indexes as nothing and the model can only answer "I don't know". With
    # this on, images are also described by a vision model and that caption is
    # indexed alongside the OCR text. Needs a vision-capable model: the default
    # chat model (llama-3.3-70b-versatile) is text-only.
    vision_enabled: bool = True
    groq_vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    vision_max_tokens: int = 400
    # Long edge to downscale to before upload: keeps requests under Groq's
    # base64 size limit and cuts latency. Phone photos are far larger.
    vision_max_pixels: int = 1024

    # --- Vector store (Qdrant) ------------------------------------------
    # Leave qdrant_url empty for embedded on-disk mode (uses qdrant_path).
    # Set it (e.g. http://localhost:6333) to talk to a Qdrant server, which
    # lifts the single-process lock and enables a payload index on user_id.
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_path: str = "./data/qdrant"
    qdrant_collection: str = "rag_documents"

    # --- Application database (PostgreSQL: users + chat history) ---------
    # libpq connection URI. `docker compose up -d postgres` starts a local
    # server matching this default. For a managed/remote database, set
    # DATABASE_URL in .env (add ?sslmode=require for TLS).
    database_url: str = "postgresql://raguser:ragpass@localhost:5434/ragdb"

    # --- Object storage (MinIO / S3-compatible) -------------------------
    # Durable per-user storage for uploaded document originals and JSON
    # snapshots of every chat, keyed under users/{user_id}/ (see
    # rag_app/storage.py). PostgreSQL stays the live chat store; MinIO is the
    # durable copy. Off by default so the app runs with nothing else present:
    # when disabled, originals are discarded after ingest (the old behaviour)
    # and chats live only in PostgreSQL. `docker compose up -d minio` starts one
    # locally that matches the defaults below.
    storage_enabled: bool = False
    minio_endpoint: str = "localhost:9000"   # host:port, no scheme
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "rag-storage"
    minio_secure: bool = False               # True for HTTPS (any remote deployment)
    minio_region: str = ""

    # --- Web layer (FastAPI auth tokens + CORS) -------------------------
    # Used to sign JWT session tokens for the React UI. CHANGE THIS in
    # production via JWT_SECRET in .env — a leaked secret lets anyone forge
    # a token for any user. If left at the default, a warning is logged.
    jwt_secret: str = "dev-insecure-change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7  # 7 days
    # Comma-separated origins allowed to call the API (the Vite dev server).
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- Password hashing ------------------------------------------------
    pbkdf2_iterations: int = 200_000

    # --- Chunking --------------------------------------------------------
    chunk_size: int = 1000          # characters per chunk
    chunk_overlap: int = 150        # overlap between consecutive chunks

    # --- Retrieval / generation -----------------------------------------
    top_k: int = 4                  # chunks retrieved per query
    # Drop retrieved chunks scoring below this (0.0 = keep all). With
    # reranking enabled this is applied to the rerank score.
    retrieval_score_threshold: float = 0.0
    # How many prior messages of a conversation to feed the model for
    # follow-up understanding and answer continuity.
    history_turns: int = 6
    max_output_tokens: int = 1024
    temperature: float = 0.2
    log_queries: bool = True        # log per-query timings + token usage

    # --- Web layer: security & limits -----------------------------------
    # In dev mode the server runs with the insecure default JWT_SECRET (a
    # warning is logged). Set DEV_MODE=false in production so startup fails
    # fast unless a strong JWT_SECRET is configured.
    dev_mode: bool = True
    max_upload_mb: int = 25         # reject uploads larger than this

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS origins as a clean list (drops blanks)."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def ensure_data_dirs(self) -> None:
        """Create parent directories for the local data stores."""
        Path(self.qdrant_path).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    settings = Settings()
    settings.ensure_data_dirs()
    return settings
