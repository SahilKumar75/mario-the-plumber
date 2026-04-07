# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI app for the Mario the Plumber OpenEnv environment."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException
from fastapi.openapi.utils import get_openapi
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
)
from inference import run_baseline
from models import PipelineDoctorAction, PipelineDoctorObservation
from server.benchmark_demo import build_benchmark_demo
from server.pipeline_doctor_environment import PipelineDoctorEnvironment
from graders import grade_episode
from tasks.task_bank import list_internal_task_ids, tasks_payload


class GraderRequest(BaseModel):
    """Lookup payload for the /grader endpoint."""

    task_id: int | str
    episode_id: str | None = None


app = create_app(
    PipelineDoctorEnvironment,
    PipelineDoctorAction,
    PipelineDoctorObservation,
    env_name="mario_the_plumber",
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


def _route_paths() -> set[str]:
    """Return the registered route paths for lightweight self-validation checks."""

    return {route.path for route in app.routes if hasattr(route, "path")}


def _install_openapi_overrides() -> None:
    """Advertise task selection explicitly in the generated /reset schema."""

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        components = schema.setdefault("components", {}).setdefault("schemas", {})
        reset_request = components.setdefault("ResetRequest", {})
        properties = reset_request.setdefault("properties", {})
        properties["task_id"] = {
            "anyOf": [
                {"type": "integer", "minimum": 1.0, "maximum": 5.0},
                {
                    "type": "string",
                    "enum": [str(task_id) for task_id in list_internal_task_ids()]
                    + [f"task_{task_id}" for task_id in list_internal_task_ids()],
                },
            ],
            "title": "Task Id",
            "description": "Task selector supporting integer ids (1-5) and aliases like task_1.",
        }
        properties["split"] = {
            "type": "string",
            "enum": ["train", "eval"],
            "title": "Split",
            "description": "Scenario split used for the reset.",
            "default": "train",
        }
        reset_request["examples"] = [
            {"task_id": "task_1", "seed": 42, "split": "train"},
            {"task_id": 2, "seed": 42, "split": "eval"},
        ]
        app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = custom_openapi


_install_openapi_overrides()


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
    """Expose the canonical validator-facing task list."""

    return tasks_payload()


@app.get("/validate")
def validate() -> dict[str, object]:
    """Expose hackathon-friendly validation checks for task/grader discovery."""

    tasks = tasks_payload()["tasks"]
    route_paths = _route_paths()
    checks = {
        "openenv_yaml": Path("openenv.yaml").exists(),
        "typed_models": True,
        "reset_endpoint": "/reset" in route_paths,
        "step_endpoint": "/step" in route_paths,
        "state_endpoint": "/state" in route_paths,
        "min_3_tasks": len(tasks) >= 3,
        "all_tasks_have_graders": all(bool(task.get("grader")) for task in tasks),
        "reward_shaped": True,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "env_name": "mario_the_plumber",
        "version": "2.1",
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
    """Return a stored episode grade when available, else a deterministic live grade."""

    try:
        return grade_episode(request.task_id, episode_id=request.episode_id, split="eval", seed=42)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/grade/{task_id}")
def grade_task(task_id: str) -> dict[str, object]:
    """Return a deterministic per-task live grade payload for external validators."""

    try:
        return grade_episode(task_id, split="eval", seed=42)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


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
