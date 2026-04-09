# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Custom FastAPI app for the Mario the Plumber OpenEnv environment."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.openapi.utils import get_openapi
from fastapi.responses import RedirectResponse
from openenv.core.env_server.http_server import serialize_observation
from openenv.core.env_server.types import EnvironmentMetadata, HealthResponse, HealthStatus, SchemaResponse
from pydantic import BaseModel

from benchmark.api_payloads import (
    VALIDATOR_TASK_IDS,
    adaptation_payload,
    benchmark_metadata_payload,
    benchmark_profiles_payload,
    benchmark_runs_payload,
    benchmark_tasks_payload,
)
from debug_trace import debug_log
from inference import run_baseline
from models import PipelineDoctorAction, PipelineDoctorObservation, PipelineDoctorState
from graders import debug_grade_payload, validator_grade_payload
from server.pipeline_doctor_environment import PipelineDoctorEnvironment
from tasks.definitions import list_internal_task_ids, task_payloads, tasks_payload

_ENV = PipelineDoctorEnvironment()
_ENV_LOCK = Lock()
debug_log("server_app_init", validator_task_ids=list(VALIDATOR_TASK_IDS))


class ResetRequest(BaseModel):
    """Reset payload for the shared-environment HTTP server."""

    seed: int | None = None
    episode_id: str | None = None
    task_id: int | str = "task_1"
    split: str = "train"


class GraderRequest(BaseModel):
    """Lookup payload for the /grader endpoint."""

    task_id: int | str = "task_1"
    episode_id: str | None = None


class BaselineRequest(BaseModel):
    """Request body for the /baseline endpoint."""

    seed: int = 42
    split: str = "train"
    policy_mode: str = "hybrid"


app = FastAPI(
    title="OpenEnv Environment HTTP API",
    version="2.1.0",
    description="HTTP API for interacting with the Mario the Plumber environment.",
    docs_url=None,
    redoc_url=None,
    openapi_url="/openapi.json",
)


def _route_paths() -> set[str]:
    return {route.path for route in app.routes if hasattr(route, "path")}


def _tasks() -> list[dict[str, object]]:
    return task_payloads()


def _validator_grade_payload(payload: dict[str, object]) -> dict[str, float]:
    score = float(payload.get("score", 0.0))
    reward = float(payload.get("reward", score))
    return {
        "score": score,
        "reward": reward,
    }


def _install_openapi_overrides() -> None:
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
                {"type": "integer", "minimum": 1.0, "maximum": float(max(VALIDATOR_TASK_IDS))},
                {
                    "type": "string",
                    "enum": [str(task_id) for task_id in list_internal_task_ids()]
                    + [f"task_{task_id}" for task_id in list_internal_task_ids()],
                },
            ],
            "title": "Task Id",
            "description": "Task selector supporting integer ids and aliases like task_1.",
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


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/metadata", status_code=307)


@app.get("/web", include_in_schema=False)
def web_root() -> RedirectResponse:
    return RedirectResponse(url="/", status_code=307)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    debug_log("health_request")
    return HealthResponse(status=HealthStatus.HEALTHY)


@app.get("/metadata", response_model=EnvironmentMetadata)
def metadata() -> EnvironmentMetadata:
    return _ENV.get_metadata()


@app.get("/schema", response_model=SchemaResponse)
def schema() -> SchemaResponse:
    return SchemaResponse(
        action=PipelineDoctorAction.model_json_schema(),
        observation=PipelineDoctorObservation.model_json_schema(),
        state=PipelineDoctorState.model_json_schema(),
    )


@app.post("/reset")
def reset(request: ResetRequest = Body(default_factory=ResetRequest)) -> dict[str, object]:
    debug_log(
        "reset_request",
        task_id=request.task_id,
        split=request.split,
        seed=request.seed,
        episode_id=request.episode_id,
    )
    with _ENV_LOCK:
        observation = _ENV.reset(**request.model_dump())
        debug_log(
            "reset_response",
            task_id=_ENV.state.task_id,
            episode_id=_ENV.state.episode_id,
            current_score=_ENV.state.current_score,
            done=_ENV.state.done,
        )
        return serialize_observation(observation)


@app.post("/step")
async def step(request: Request) -> dict[str, object]:
    payload = await request.json()
    debug_log("step_request", payload=payload)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="step payload must be a JSON object")

    action_blob = payload.get("action", payload)
    if not isinstance(action_blob, dict):
        raise HTTPException(status_code=422, detail="action payload must be a JSON object")

    try:
        action = PipelineDoctorAction.model_validate(action_blob)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    with _ENV_LOCK:
        observation = _ENV.step(action)
        debug_log(
            "step_response",
            task_id=_ENV.state.task_id,
            episode_id=_ENV.state.episode_id,
            action_id=action.action_id,
            done=observation.done,
            reward=observation.reward,
            current_score=observation.current_score,
            action_result=observation.action_result,
        )
        return serialize_observation(observation)


@app.get("/state")
def state() -> dict[str, Any]:
    with _ENV_LOCK:
        debug_log("state_request", task_id=_ENV.state.task_id, episode_id=_ENV.state.episode_id)
        return _ENV.state.model_dump()


@app.get("/tasks")
def get_tasks() -> dict[str, object]:
    debug_log("tasks_request")
    return tasks_payload()


@app.get("/validate")
def validate() -> dict[str, object]:
    tasks = _tasks()
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
    payload = {
        "valid": all(checks.values()),
        "checks": checks,
        "env_name": "mario_the_plumber",
        "version": "2.1",
    }
    debug_log("validate_request", valid=payload["valid"], checks=checks)
    return payload


@app.get("/benchmark-metadata")
def get_benchmark_metadata() -> dict[str, object]:
    return benchmark_metadata_payload()


@app.get("/benchmark/metadata")
def get_runtime_metadata() -> dict[str, object]:
    return get_benchmark_metadata()


@app.get("/benchmark/tasks")
def get_benchmark_tasks() -> dict[str, object]:
    return benchmark_tasks_payload()


@app.get("/benchmark/profiles")
def get_benchmark_profiles() -> dict[str, object]:
    return benchmark_profiles_payload()


@app.get("/benchmark/runs")
def get_benchmark_runs() -> dict[str, object]:
    return benchmark_runs_payload()


@app.get("/benchmark/adaptation")
def get_benchmark_adaptation() -> dict[str, object]:
    return adaptation_payload()


@app.post("/grader")
def grader(request: GraderRequest = Body(default_factory=GraderRequest)) -> dict[str, object]:
    try:
        payload = validator_grade_payload(
            request.task_id,
            episode_id=request.episode_id,
            split="eval",
            seed=42,
        )
        debug_log("grader_request", task_id=request.task_id, episode_id=request.episode_id, payload=payload)
        return payload
    except ValueError as exc:
        debug_log("grader_error", task_id=request.task_id, episode_id=request.episode_id, error=str(exc))
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/grader")
def grader_get(task_id: str = Query(default="task_1"), episode_id: str | None = Query(default=None)) -> dict[str, object]:
    try:
        payload = validator_grade_payload(
            task_id,
            episode_id=episode_id,
            split="eval",
            seed=42,
        )
        debug_log("grader_get_request", task_id=task_id, episode_id=episode_id, payload=payload)
        return payload
    except ValueError as exc:
        debug_log("grader_get_error", task_id=task_id, episode_id=episode_id, error=str(exc))
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/grade/{task_id}")
def grade_task(task_id: str) -> dict[str, object]:
    try:
        payload = validator_grade_payload(task_id, split="eval", seed=42)
        debug_log("grade_task_request", task_id=task_id, payload=payload)
        return payload
    except ValueError as exc:
        debug_log("grade_task_error", task_id=task_id, error=str(exc))
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/grader/debug")
def grader_debug(task_id: str = Query(default="task_1"), episode_id: str | None = Query(default=None)) -> dict[str, object]:
    try:
        payload = debug_grade_payload(
            task_id,
            episode_id=episode_id,
            split="eval",
            seed=42,
        )
        debug_log("grader_debug_request", task_id=task_id, episode_id=episode_id, payload=payload)
        return payload
    except ValueError as exc:
        debug_log("grader_debug_error", task_id=task_id, episode_id=episode_id, error=str(exc))
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/grade-debug/{task_id}")
def grade_task_debug(task_id: str) -> dict[str, object]:
    try:
        payload = debug_grade_payload(task_id, split="eval", seed=42)
        debug_log("grade_task_debug_request", task_id=task_id, payload=payload)
        return payload
    except ValueError as exc:
        debug_log("grade_task_debug_error", task_id=task_id, error=str(exc))
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/baseline")
def baseline(request: BaselineRequest) -> dict[str, object]:
    debug_log("baseline_request", seed=request.seed, split=request.split, policy_mode=request.policy_mode)
    payload = run_baseline(
        seed=request.seed,
        split=request.split,
        policy_mode=request.policy_mode,
    )
    debug_log("baseline_response", average_score=payload["average_score"], status=payload["status"])
    return payload


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
