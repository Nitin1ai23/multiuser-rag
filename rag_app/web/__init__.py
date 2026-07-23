"""Web layer: a FastAPI backend that exposes the RAG core over HTTP.

This package is a thin adapter. All real logic — auth, per-user vector
isolation, chat history, RAG — lives in the shared ``rag_app`` core and is
reused unchanged by both the PyQt5 desktop UI and this web API.
"""
