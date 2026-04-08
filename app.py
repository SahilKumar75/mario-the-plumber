from __future__ import annotations

from typing import Any

from fastapi import Body, FastAPI, Request
import gradio as gr

from server.app import (
    _ENV,
    BaselineRequest,
    GraderRequest,
    ResetRequest,
    app as _unused_api_app,
    baseline as api_baseline,
    get_benchmark_adaptation,
    get_benchmark_metadata,
    get_benchmark_profiles,
    get_benchmark_runs,
    get_benchmark_tasks,
    grader as api_grader,
    grade_task as api_grade_task,
    health as api_health,
    main as api_main,
    metadata as api_metadata,
    reset as api_reset,
    schema as api_schema,
    state as api_state,
    step as api_step,
    get_tasks as api_tasks,
    validate as api_validate,
)
from server.benchmark_demo import create_space_demo

demo = create_space_demo(_ENV)
app = FastAPI(title="Mario the Plumber Space")
app.mount("/api", _unused_api_app)


@app.get("/health")
def health():
    return api_health()


@app.get("/metadata")
def metadata():
    return api_metadata()


@app.get("/schema")
def schema():
    return api_schema()


@app.post("/reset")
def reset(request: ResetRequest = Body(default_factory=ResetRequest)):
    return api_reset(request)


@app.post("/step")
async def step(request: Request):
    return await api_step(request)


@app.get("/state")
def state() -> dict[str, Any]:
    return api_state()


@app.get("/tasks")
def tasks():
    return api_tasks()


@app.get("/validate")
def validate():
    return api_validate()


@app.get("/benchmark-metadata")
def benchmark_metadata():
    return get_benchmark_metadata()


@app.get("/benchmark/metadata")
def benchmark_runtime_metadata():
    return get_benchmark_metadata()


@app.get("/benchmark/tasks")
def benchmark_tasks():
    return get_benchmark_tasks()


@app.get("/benchmark/profiles")
def benchmark_profiles():
    return get_benchmark_profiles()


@app.get("/benchmark/runs")
def benchmark_runs():
    return get_benchmark_runs()


@app.get("/benchmark/adaptation")
def benchmark_adaptation():
    return get_benchmark_adaptation()


@app.post("/grader")
def grader(request: GraderRequest = Body(default_factory=GraderRequest)):
    return api_grader(request)


@app.get("/grade/{task_id}")
def grade_task(task_id: str):
    return api_grade_task(task_id)


@app.post("/baseline")
def baseline(request: BaselineRequest):
    return api_baseline(request)

app = gr.mount_gradio_app(app, demo, path="/")


def main() -> None:
    api_main()


__all__ = ["app", "demo", "main"]
