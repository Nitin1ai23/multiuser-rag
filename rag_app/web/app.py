"""FastAPI application factory for the RAG web backend.

Run it with::

    uv run --no-project uvicorn rag_app.web.app:app --reload   # dev
    uv run --no-project python -m rag_app.web.app               # prod-ish

In development the React app runs on the Vite dev server (port 5173) and calls
this API on port 8000 (CORS is configured for that). In production, build the
frontend (``npm run build``) and this server will also serve ``frontend/dist``
as a single-page app, so everything is on one origin.

Note: embedded Qdrant locks its storage directory, so run a SINGLE worker and
do not run the PyQt5 desktop app against the same ``QDRANT_PATH`` at the same
time.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..config import get_settings
from .routes import auth, chat, documents

_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


def create_app() -> FastAPI:
    settings = get_settings()

    # Fail fast in production rather than silently signing tokens with a known
    # secret (which would let anyone forge a session for any user).
    if not settings.dev_mode and settings.jwt_secret == "dev-insecure-change-me":
        raise RuntimeError(
            "Refusing to start: JWT_SECRET is the insecure default while "
            "DEV_MODE is false. Set a strong JWT_SECRET in .env "
            '(python -c "import secrets; print(secrets.token_urlsafe(32))").'
        )

    app = FastAPI(title="Multi-user RAG API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api = "/api"
    app.include_router(auth.router, prefix=api)
    app.include_router(chat.router, prefix=api)
    app.include_router(documents.router, prefix=api)

    @app.get(f"{api}/health", tags=["meta"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # Serve the built React SPA if it exists (production single-origin mode).
    if _DIST.is_dir():
        app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="spa")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("rag_app.web.app:app", host="127.0.0.1", port=8000, reload=False)
