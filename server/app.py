# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI app for the Mario the Plumber OpenEnv environment."""

from __future__ import annotations

from fastapi import HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError("openenv-core is required to run the Mario the Plumber server.") from exc

from benchmark.api_payloads import (
    adaptation_payload,
    benchmark_metadata_payload,
    benchmark_profiles_payload,
    benchmark_runs_payload,
    benchmark_tasks_payload,
    tasks_payload,
)
from benchmark.catalog import TASK_THRESHOLDS
from benchmark.evaluation import breakdown_payload, objective_breakdown, score
from inference import run_baseline
from models import PipelineDoctorAction, PipelineDoctorObservation
from server.benchmark_demo import build_benchmark_demo
from server.pipeline_doctor_environment import EPISODE_SUMMARIES, PipelineDoctorEnvironment


class GraderRequest(BaseModel):
    """Lookup payload for the /grader endpoint."""

    task_id: int
    episode_id: str | None = None


app = create_app(
    PipelineDoctorEnvironment,
    PipelineDoctorAction,
    PipelineDoctorObservation,
    env_name="mario_the_plumber",
    # Force a single environment instance so reset/step state stays in one session.
    max_concurrent_envs=1,
    gradio_builder=build_benchmark_demo,
)


def _jsonify(value):
    """Convert numpy/pydantic-like values into plain JSON-native structures."""

    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


@app.get("/health")
def health() -> dict[str, str]:
    """Lightweight health endpoint for container readiness checks."""

    return {"status": "healthy"}


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect root requests to the benchmark web UI."""

    return RedirectResponse(url="/docs", status_code=307)


@app.get("/web", include_in_schema=False)
def web_root() -> RedirectResponse:
    """Provide a stable web path that lands on FastAPI docs in API-only mode."""

    return RedirectResponse(url="/docs", status_code=307)


@app.get("/tasks")
def get_tasks() -> dict[str, object]:
    """Expose the benchmark task list and action schema."""

    return tasks_payload()


def _preview_grade(task_id: int) -> dict[str, object]:
    """Run a deterministic task preview so graders can be validated without session state."""

    if task_id not in TASK_THRESHOLDS:
        raise HTTPException(status_code=404, detail=f"unknown task_id {task_id}")
    env = PipelineDoctorEnvironment()
    observation = env.reset(seed=42, task_id=task_id, split="train")
    preview_score = round(float(score(env)), 4)
    return {
        "task_id": task_id,
        "score": preview_score,
        "reward": round(float(observation.reward), 4),
        "success_threshold": TASK_THRESHOLDS[task_id],
        "success": bool(preview_score >= TASK_THRESHOLDS[task_id]),
        "breakdown": _jsonify(breakdown_payload(env)),
        "objective_breakdown": _jsonify(objective_breakdown(env)),
        "grader_mode": "preview",
    }


@app.get("/benchmark-metadata")
def get_benchmark_metadata() -> dict[str, object]:
    """Expose benchmark profiles, utility notes, and task metadata."""

    return benchmark_metadata_payload()


@app.get("/benchmark/metadata")
def get_runtime_metadata() -> dict[str, object]:
    """Expose benchmark metadata under the dedicated benchmark route namespace."""

    return get_benchmark_metadata()


@app.get("/benchmark/tasks")
def get_benchmark_tasks() -> dict[str, object]:
    """Expose task cards, objective weights, and formal task specs."""

    return benchmark_tasks_payload()


@app.get("/benchmark/profiles")
def get_benchmark_profiles() -> dict[str, object]:
    """Expose train/eval profile families and synthetic-data notes."""

    return benchmark_profiles_payload()


@app.get("/benchmark/runs")
def get_benchmark_runs() -> dict[str, object]:
    """Expose the latest benchmark ladder artifact, if available."""

    return benchmark_runs_payload()


@app.get("/benchmark/adaptation")
def get_benchmark_adaptation() -> dict[str, object]:
    """Expose the latest adaptation artifact, if available."""

    return adaptation_payload()


@app.post("/grader")
def grader(request: GraderRequest) -> dict[str, object]:
    """Return the latest stored episode summary for a finished episode."""

    if not request.episode_id:
        return _preview_grade(request.task_id)

    payload = EPISODE_SUMMARIES.get(request.episode_id)
    if payload is None:
        preview = _preview_grade(request.task_id)
        return {
            **preview,
            "episode_id": request.episode_id,
            "error": "episode_id not found; returned deterministic preview grade instead",
        }
    return payload


@app.get("/grade/{task_id}")
def grade_task(task_id: int) -> dict[str, object]:
    """Return a deterministic per-task grade payload for external validators."""

    return _preview_grade(task_id)


class BaselineRequest(BaseModel):
    """Request body for the /baseline endpoint."""

    seed: int = 42
    split: str = "train"
    policy_mode: str = "hybrid"


@app.post("/baseline")
def baseline(request: BaselineRequest) -> dict[str, object]:
    """Run the local submission baseline with configurable seed, split, and policy mode."""

    return run_baseline(
        seed=request.seed,
        split=request.split,
        policy_mode=request.policy_mode,
    )


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the development server directly."""

    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
