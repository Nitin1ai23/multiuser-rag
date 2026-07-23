# Multi-user RAG App

A multi-user RAG app where each user has a private account and can only ever
see and query **their own** documents. Each login starts a fresh chat while
keeping the user's ingested documents available. Ships with two
frontends over the same isolated core: a **PyQt5 desktop app** and a
**web UI** (FastAPI backend + React/Vite frontend).

- **Desktop UI:** PyQt5 (login / sign-up / forgot-password → per-user chat + documents)
- **Web UI:** React + Vite SPA talking to a FastAPI backend (`rag_app/web/`)
- **Retrieval:** Qdrant (embedded on-disk by default; point at a server with `QDRANT_URL`)
- **Embeddings:** NVIDIA NIM (`nv-embedqa`) — Groq has no embeddings API
- **Reranking:** optional NVIDIA NIM cross-encoder re-scores candidates for precision
  (best-effort, falls back to vector order; off by default — see `NIM_RERANK_URL` in `.env.example`)
- **Generation:** Groq (`llama-3.3-70b-versatile`), **streamed** token-by-token in the web UI
- **Graceful fallback:** a question the documents don't cover still gets a short general-knowledge
  answer (flagged as such, never cited) plus a prompt to upload a relevant document
- **Conversational:** follow-up questions are condensed against recent history before retrieval
- **Storage:** optional MinIO (S3-compatible) keeps a per-user copy of every chat and every
  uploaded original under `users/{user_id}/` — off by default (see [Object storage](#object-storage-minio))
- **Accounts:** PostgreSQL, salted PBKDF2-HMAC-SHA256 password hashing; self-service account deletion
- **Password reset:** security question (no email server required)
- **Hardening:** per-IP auth rate limiting, revocable JWT sessions (logout), upload size cap,
  background ingestion with progress polling, and fail-fast on an insecure secret in production

## How isolation works

Every user gets a UUID at sign-up. That id is the isolation key:

- **Vectors:** every chunk stored in Qdrant carries the owner's `user_id`, and
  *every* search/list/delete is filtered by it (`rag_app/vectorstore.py`). There
  is no code path that returns points across users.
- **Chat messages:** stored in PostgreSQL keyed by `user_id`
  (`rag_app/chat/history.py`), scoped to one user, and cleared on successful
  auth so each login opens a new chat without deleting vectors.
- A `RAGPipeline` is bound to a single `user_id` at construction
  (`rag_app/rag.py`), so the UI cannot accidentally cross tenants.
- **Stored objects:** every MinIO key begins with `users/{user_id}/`
  (`rag_app/storage.py`), so a key is built from the caller's own id and cannot
  name another user's object. Ids containing `/` or `..` are rejected rather
  than escaped, so no id can walk out of its owner's prefix.

The `smoke_test.py` script asserts that user A cannot retrieve user B's chunks
even when querying with B's own vector.

## Setup

```bash
cp .env.example .env        # then fill in NVIDIA_API_KEY and GROQ_API_KEY
uv venv --python 3.13
uv pip install PyQt5 qdrant-client groq openai pydantic-settings python-dotenv pypdf httpx minio "psycopg[binary]"
# richer ingestion (Word/Excel/PowerPoint + image OCR):
uv pip install python-docx openpyxl python-pptx Pillow pytesseract   # OCR also needs the Tesseract binary on PATH
# web layer:
uv pip install fastapi "uvicorn[standard]" PyJWT python-multipart
# or, all at once (incl. dev tools for tests/lint):
uv pip install -e ".[dev]"
```

Get keys: NVIDIA → https://build.nvidia.com · Groq → https://console.groq.com/keys

### Database (PostgreSQL)

Users, conversations, and chat history live in PostgreSQL. Start a local server
with the bundled compose file (published on host port **5434** so it never
clashes with a Postgres you may already run on 5432):

```bash
docker compose up -d postgres    # postgres:16, data under ./data/postgres
```

The default `DATABASE_URL` in `.env.example`
(`postgresql://raguser:ragpass@localhost:5434/ragdb`) already matches it, and the
app creates its tables on first connect. Point at a managed/remote database by
setting `DATABASE_URL` (append `?sslmode=require` for TLS).

## Run — desktop app

```bash
uv run --no-project python main.py     # launch the PyQt5 app
uv run --no-project python smoke_test.py   # offline auth + isolation checks (no keys)
```

## Run — web app

Two processes in dev. The Vite dev server proxies `/api` to the backend, so the
browser sees one origin.

```bash
# 1. backend (FastAPI) on http://127.0.0.1:8000
uv run --no-project uvicorn rag_app.web.app:app --reload

# 2. frontend (Vite) on http://localhost:5173
cd frontend && npm install && npm run dev
```

For a single-origin production-style run, build the frontend first — FastAPI
then serves `frontend/dist` itself, so only the backend needs to run:

```bash
cd frontend && npm run build && cd ..
uv run --no-project python -m rag_app.web.app   # serves SPA + API on :8000
```

Set a strong `JWT_SECRET` in `.env` before exposing the server (generate one
with `python -c "import secrets; print(secrets.token_urlsafe(32))"`) and set
`DEV_MODE=false` — the backend then **refuses to start** while the secret is
still the insecure default, instead of only warning.

> Embedded Qdrant locks its storage directory, so run a **single** backend
> worker and don't point the desktop app and the web backend at the same
> `QDRANT_PATH` simultaneously. To run multiple workers, set `QDRANT_URL` to a
> Qdrant server — that lifts the lock and enables a `user_id` payload index.

## Object storage (MinIO)

Off by default — the app runs exactly as before without it. Turn it on and every
chat and every uploaded document gets a durable copy under its owner's prefix:

```
users/{user_id}/chats/{conversation_id}.json           snapshot of one chat
users/{user_id}/chats/{conversation_id}/docs/{file}     the uploaded original
```

```bash
docker compose up -d        # MinIO on :9000, console on :9001 (minioadmin/minioadmin)
# then in .env:
STORAGE_ENABLED=true
```

Two things to know about what this does and doesn't change:

- **Chats:** PostgreSQL stays the live store — every read, the sidebar, and
  follow-up history still come from it, so nothing gets slower. MinIO receives a
  JSON snapshot after each change, uploaded on a background thread
  (`rag_app/chat/mirror.py`). If MinIO is down, chat keeps working and the
  archive silently falls behind; it catches up on the next message, since each
  snapshot is the whole conversation rather than a delta.
- **Documents:** this is the *only* copy of an uploaded original. Ingestion
  previously discarded the file once it was chunked, so with storage off there
  is still nothing to download later. Chunks and vectors stay in Qdrant either
  way. Retrieve an original with
  `GET /api/documents/download?source=…&conversation_id=…`; a member of an
  uploaded `.zip` has none of its own (the archive is stored whole).

Deletion propagates: removing a document deletes its object, deleting a chat
deletes its snapshot *and* its documents, and deleting an account purges
`users/{user_id}/` entirely.

The defaults above are dev credentials on localhost. For any real deployment set
real keys and `MINIO_SECURE=true`. Any S3-compatible endpoint works.

## Tests

```bash
uv pip install -e ".[dev]"
uv run pytest -q          # auth, chunking, per-user isolation, rate limiting
uv run ruff check .
```

No API keys are needed: the tests run against a throwaway Postgres database
(`ragdb_test`, created automatically — start Postgres with `docker compose up -d
postgres`) + embedded Qdrant and
never call NIM/Groq. No MinIO either — storage is disabled in the test env and
the storage tests run against a fake store. CI
(`.github/workflows/ci.yml`) runs ruff + pytest on push.

## Usage

1. **Create account** — username, email, password, and a security question/answer.
2. **Documents tab** — upload PDFs / .txt / .md, Word (.docx), Excel (.xlsx),
   and PowerPoint (.pptx); they're chunked, embedded, and
   stored under your account. Ingestion runs in the background (with a size cap);
   re-uploading the same filename replaces the previous version instead of
   duplicating it.
   **Images** are handled two ways at once: a vision model describes what the
   image *shows*, and Tesseract OCRs any text *in* it — both are indexed, so a
   photo or chart with no text in it is still answerable. This costs one extra
   Groq call per image; set `VISION_ENABLED=false` for OCR only. Note that
   `GROQ_VISION_MODEL` must be vision-capable (`GROQ_MODEL` is text-only), and
   images inside a `.zip` are OCR'd only — they aren't captioned.
3. **Chat tab** — ask questions; answers stream in token-by-token (web UI),
   generated by Groq from your retained chunks, with sources cited.
   Follow-up questions are understood in context of the current conversation.
   Ask something your documents *don't* cover and it won't dead-end at "I don't
   know": it says your documents don't cover it, answers briefly from general
   knowledge (marked as such, never given a citation), and invites you to upload
   a document for a detailed, cited answer.
4. **Forgot password** — enter your username/email, answer your security
   question, set a new password.
5. **Account** — "Delete account" (web UI) permanently removes your documents,
   chats, vectors, and stored objects after a password confirmation.

## Project layout

```
rag_app/
  config.py          settings (NIM, Groq, Qdrant, MinIO, Postgres URL, limits)
  embeddings.py      NIM embedder (shared singleton)
  reranker.py        NIM cross-encoder reranker (best-effort)
  ingest.py          load + chunk pdf/txt/md/docx/xlsx/pptx + image OCR
  vectorstore.py     Qdrant wrapper — per-user filtered (embedded or server)
  storage.py         MinIO object store — keys namespaced by user_id (optional)
  llm.py             Groq chat provider (blocking + streaming)
  rag.py             RAGPipeline(user_id) — ingest + history-aware query/stream,
                     prompts incl. the no-context general-knowledge fallback
  auth/
    db.py            PostgreSQL schema + per-thread connection (psycopg)
    service.py       signup / login / security-question reset
  chat/
    history.py       per-user message history
    conversations.py per-user chat sessions
    mirror.py        archives each chat to object storage (background, ordered)
  ui/
    login_window.py  login / signup / forgot screens
    main_window.py   per-user chat + documents
    workers.py       QThread workers (init / ingest / query)
    app.py           window orchestration
  web/               FastAPI backend (reuses the core unchanged)
    app.py           app factory: CORS, routers, serves built SPA
    security.py      JWT session tokens (+ jti revocation) + current_user
    ratelimit.py     per-IP fixed-window limiter for auth endpoints
    ingest_jobs.py   in-memory background ingestion job registry
    schemas.py       request/response models
    routes/          auth.py · chat.py (incl. SSE stream) · documents.py
tests/               pytest: auth, chunking, isolation, rate limits, storage, prompts
main.py              desktop entry point
docker-compose.yml   Postgres + MinIO for local development
frontend/            React + Vite web UI
  src/
    api.js           fetch wrapper (JWT in localStorage)
    App.jsx          session routing (auth screen vs app)
    components/       AuthScreen · ChatApp · Sidebar · Conversation
```

Both UIs are thin: `rag_app/web/` and `rag_app/ui/` each just call into the
shared, user-scoped core (`AuthService`, `RAGPipeline`, `ChatHistory`,
`QdrantStore`), so per-user isolation holds identically across both.
