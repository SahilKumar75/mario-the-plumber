from __future__ import annotations

from fastapi import FastAPI
import gradio as gr

from server.app import _ENV, app as api_app, main as api_main
from server.benchmark_demo import create_space_demo

demo = create_space_demo(_ENV)
app = FastAPI(title="Mario the Plumber Space")
app.mount("/api", api_app)
app = gr.mount_gradio_app(app, demo, path="/")


def main() -> None:
    api_main()


__all__ = ["app", "demo", "main"]
