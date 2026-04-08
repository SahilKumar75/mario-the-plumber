from __future__ import annotations

import gradio as gr

from server.app import _ENV, app as api_app, main as api_main
from server.benchmark_demo import create_space_demo

demo = create_space_demo(_ENV)
app = gr.mount_gradio_app(api_app, demo, path="/")


def main() -> None:
    api_main()


__all__ = ["app", "demo", "main"]
