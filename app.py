"""Root app entrypoint mirroring the FastAPI application exported by server.app."""

from server.app import app, main

__all__ = ["app", "main"]
