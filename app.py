from __future__ import annotations

from typing import Any

from debug_trace import debug_log
from fastapi import Body, FastAPI, Query, Request
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
    grader_debug as api_grader_debug,
    grade_task as api_grade_task,
    grade_task_debug as api_grade_task_debug,
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
debug_log("space_app_init", mounted_api_prefix="/api")


@app.get("/health")
def health():
    debug_log("space_health_request")
    return api_health()


@app.get("/metadata")
def metadata():
    debug_log("space_metadata_request")
    return api_metadata()


@app.get("/schema")
def schema():
    debug_log("space_schema_request")
    return api_schema()


@app.post("/reset")
def reset(request: ResetRequest = Body(default_factory=ResetRequest)):
    debug_log("space_reset_request", task_id=request.task_id, split=request.split, seed=request.seed)
    return api_reset(request)


@app.post("/step")
async def step(request: Request):
    debug_log("space_step_request")
    return await api_step(request)


@app.get("/state")
def state() -> dict[str, Any]:
    debug_log("space_state_request")
    return api_state()


@app.get("/tasks")
def tasks(request: Request):
    debug_log("space_tasks_request")
    return api_tasks(request)


@app.get("/validate")
def validate():
    debug_log("space_validate_request")
    return api_validate()


@app.get("/benchmark-metadata")
def benchmark_metadata():
    debug_log("space_benchmark_metadata_request")
    return get_benchmark_metadata()


@app.get("/benchmark/metadata")
def benchmark_runtime_metadata():
    debug_log("space_benchmark_runtime_metadata_request")
    return get_benchmark_metadata()


@app.get("/benchmark/tasks")
def benchmark_tasks():
    debug_log("space_benchmark_tasks_request")
    return get_benchmark_tasks()


@app.get("/benchmark/profiles")
def benchmark_profiles():
    debug_log("space_benchmark_profiles_request")
    return get_benchmark_profiles()


@app.get("/benchmark/runs")
def benchmark_runs():
    debug_log("space_benchmark_runs_request")
    return get_benchmark_runs()


@app.get("/benchmark/adaptation")
def benchmark_adaptation():
    debug_log("space_benchmark_adaptation_request")
    return get_benchmark_adaptation()


@app.post("/grader")
def grader(request: GraderRequest = Body(default_factory=GraderRequest)):
    debug_log("space_grader_request", task_id=request.task_id, episode_id=request.episode_id)
    return api_grader(request)


@app.get("/grader")
def grader_get(task_id: str = Query(default="task_1"), episode_id: str | None = Query(default=None)):
    debug_log("space_grader_get_request", task_id=task_id, episode_id=episode_id)
    return api_grader(GraderRequest(task_id=task_id, episode_id=episode_id))


@app.get("/grade/{task_id}")
def grade_task(task_id: str):
    debug_log("space_grade_task_request", task_id=task_id)
    return api_grade_task(task_id)


@app.get("/grader/debug")
def grader_debug(task_id: str = Query(default="task_1"), episode_id: str | None = Query(default=None)):
    debug_log("space_grader_debug_request", task_id=task_id, episode_id=episode_id)
    return api_grader_debug(task_id=task_id, episode_id=episode_id)


@app.get("/grade-debug/{task_id}")
def grade_task_debug(task_id: str):
    debug_log("space_grade_task_debug_request", task_id=task_id)
    return api_grade_task_debug(task_id)


@app.post("/baseline")
def baseline(request: BaselineRequest):
    debug_log("space_baseline_request", seed=request.seed, split=request.split, policy_mode=request.policy_mode)
    return api_baseline(request)

app = gr.mount_gradio_app(app, demo, path="/")


def main() -> None:
    api_main()


__all__ = ["app", "demo", "main"]
