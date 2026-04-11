"""Openenv-style compatibility server for Mario ETL runtime.

This file adds a root-level server entrypoint similar to openenv_hackthon while
delegating all behavior to Mario's existing ETL environment implementation.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urljoin

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Make this module package-like so imports such as "server.app" keep working.
__path__ = [str(Path(__file__).with_name("server"))]  # type: ignore[var-annotated]

from benchmark.task_ids import parse_task_id, public_task_id
from models import PipelineDoctorAction
from server.pipeline_doctor_environment import PipelineDoctorEnvironment
from tasks.definitions import list_tasks


_COMPAT_TO_INTERNAL = {
    "alert_prioritization": "task_1",
    "threat_detection": "task_2",
    "incident_response": "task_3",
}

_INTERNAL_TO_COMPAT = {
    "task_1": "alert_prioritization",
    "task_2": "threat_detection",
    "task_3": "incident_response",
}

_DEFAULT_SPLIT = "train"
_ENVS: dict[str, PipelineDoctorEnvironment] = {}
_ENV_LOCK = Lock()


class ResetRequest(BaseModel):
    task_id: str = "alert_prioritization"
    scenario_index: int = 0
    seed: int | None = None
    split: str = Field(default=_DEFAULT_SPLIT, pattern="^(train|eval)$")


class StepRequest(BaseModel):
    task_id: str = "alert_prioritization"
    scenario_index: int = 0
    action: Any


class GraderRequest(BaseModel):
    task_id: str = "alert_prioritization"
    episode_id: str | None = None
    seed: int = 42
    split: str = Field(default="eval", pattern="^(train|eval)$")


# Explicit rebuild avoids deferred annotation edge cases when module is loaded
# via non-standard import paths during compatibility smoke checks.
ResetRequest.model_rebuild()
StepRequest.model_rebuild()
GraderRequest.model_rebuild()


app = FastAPI(
    title="Mario the Plumber Compatibility API",
    description=(
        "Openenv-style wrapper API. Underlying logic remains Mario ETL "
        "pipeline repair runtime."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _normalize_task_ref(task_id: str) -> tuple[int, str]:
    normalized = str(task_id).strip().lower().replace("-", "_")
    mapped = _COMPAT_TO_INTERNAL.get(normalized, normalized)
    try:
        internal_id = parse_task_id(mapped)
    except ValueError as exc:
        valid = sorted(list(_COMPAT_TO_INTERNAL) + ["task_1", "task_2", "task_3", "1", "2", "3"])
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{task_id}'. Valid examples: {valid}",
        ) from exc
    return internal_id, public_task_id(internal_id)


def _env_key(public_task_alias: str, scenario_index: int) -> str:
    return f"{public_task_alias}:{scenario_index}"


def _get_env(public_task_alias: str, scenario_index: int) -> PipelineDoctorEnvironment:
    key = _env_key(public_task_alias, scenario_index)
    with _ENV_LOCK:
        env = _ENVS.get(key)
        if env is None:
            env = PipelineDoctorEnvironment()
            _ENVS[key] = env
        return env


def _coerce_action(payload: Any) -> PipelineDoctorAction:
    if isinstance(payload, int):
        return PipelineDoctorAction(action_id=payload)
    if isinstance(payload, dict):
        try:
            return PipelineDoctorAction.model_validate(payload)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid action payload: {exc}") from exc
    raise HTTPException(status_code=422, detail="action must be an integer or object")


def _absolute_url(base_url: str, path: str) -> str:
    if path.startswith(("http://", "https://")):
        return path
    return urljoin(base_url, path.lstrip("/"))


def _public_base_url(request: Request) -> str:
    forwarded_proto = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip()
    forwarded_host = (request.headers.get("x-forwarded-host") or "").split(",")[0].strip()

    if forwarded_host:
        scheme = forwarded_proto or "https"
        return f"{scheme}://{forwarded_host}/"

    base_url = str(request.base_url)
    if "hf.space" in base_url and base_url.startswith("http://"):
        return "https://" + base_url.removeprefix("http://")
    return base_url


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "env": "Mario the Plumber ETL"}


@app.get("/tasks")
def tasks(request: Request) -> dict[str, Any]:
    base_url = _public_base_url(request)
    grader_lookup_url = _absolute_url(base_url, "/grader")
    payloads = []
    for task in list_tasks():
        compat_id = _INTERNAL_TO_COMPAT.get(task.id, task.id)
        grade_path = f"/grade/{compat_id}"
        grade_url = _absolute_url(base_url, grade_path)
        grader_http = {
            "type": "http",
            "url": grade_url,
            "method": "GET",
        }
        payloads.append(
            {
                "id": compat_id,
                "task_id": compat_id,
                "internal_task_id": task.id,
                "name": task.name,
                "difficulty": task.difficulty,
                "max_steps": task.max_steps,
                "success_threshold": task.success_threshold,
                "description": task.description,
                "grader": grade_url,
                "grader_enabled": True,
                "grade_endpoint": grade_url,
                "grader_url": grade_url,
                "grader_method": "GET",
                "graders": [grader_http],
                "grader_config": grader_http,
                "grader_function": {
                    "type": "function",
                    "endpoint": grade_url,
                },
                "episode_grader": {
                    "type": "http",
                    "url": grader_lookup_url,
                    "method": "POST",
                    "content_type": "application/json",
                    "payload_schema": {
                        "task_id": compat_id,
                        "episode_id": "<episode_id>",
                    },
                },
            }
        )
    response_payload: dict[str, Any] = {"tasks": payloads}
    for item in payloads:
        response_payload[item["id"]] = {
            "description": item["description"],
            "grader": {
                "type": "function",
                "endpoint": item["grade_endpoint"],
            },
            "grade_endpoint": item["grade_endpoint"],
            "difficulty": item["difficulty"],
        }
    return response_payload


@app.post("/reset")
def reset(req: ResetRequest | None = Body(default=None)) -> dict[str, Any]:
    if req is None:
        req = ResetRequest()
    internal_id, public_alias = _normalize_task_ref(req.task_id)
    compat_id = _INTERNAL_TO_COMPAT.get(public_alias, public_alias)
    env = _get_env(public_alias, req.scenario_index)
    with _ENV_LOCK:
        observation = env.reset(task_id=internal_id, seed=req.seed, split=req.split)
    return {
        "observation": observation.model_dump(),
        "task_id": compat_id,
        "internal_task_id": public_alias,
        "scenario_index": req.scenario_index,
    }


@app.post("/step")
def step(req: StepRequest) -> dict[str, Any]:
    _, public_alias = _normalize_task_ref(req.task_id)
    env = _get_env(public_alias, req.scenario_index)
    action = _coerce_action(req.action)
    if not getattr(env, "_scenario_meta", {}):
        raise HTTPException(status_code=400, detail="Call /reset before /step.")
    with _ENV_LOCK:
        observation = env.step(action)
    return {
        "observation": observation.model_dump(),
        "reward": observation.reward,
        "done": observation.done,
        "info": {
            "action_result": observation.action_result,
            "current_score": observation.current_score,
            "done_reason": observation.done_reason,
            "truncated": observation.truncated,
        },
    }


@app.get("/state")
def state(
    task_id: str = Query(default="alert_prioritization"),
    scenario_index: int = Query(default=0),
) -> dict[str, Any]:
    internal_id, public_alias = _normalize_task_ref(task_id)
    compat_id = _INTERNAL_TO_COMPAT.get(public_alias, public_alias)
    env = _get_env(public_alias, scenario_index)
    with _ENV_LOCK:
        current_state = env.state.model_dump()
    return {
        "task_id": compat_id,
        "internal_task_id": public_task_id(internal_id),
        "scenario_index": scenario_index,
        "state": current_state,
    }


@app.post("/grader")
def grader(req: GraderRequest | None = Body(default=None)) -> dict[str, float]:
    if req is None:
        req = GraderRequest()
    _, public_alias = _normalize_task_ref(req.task_id)
    from graders.runtime import validator_grade_payload

    return validator_grade_payload(
        public_alias,
        episode_id=req.episode_id,
        seed=req.seed,
        split=req.split,
    )


@app.get("/grader")
def grader_get(
    task_id: str = Query(default="alert_prioritization"),
    episode_id: str | None = Query(default=None),
    seed: int = Query(default=42),
    split: str = Query(default="eval"),
) -> dict[str, float]:
    return grader(
        GraderRequest(
            task_id=task_id,
            episode_id=episode_id,
            seed=seed,
            split=split,
        )
    )


@app.get("/grade/{task_id}")
def grade_task(
    task_id: str,
    episode_id: str | None = Query(default=None),
    seed: int = Query(default=42),
    split: str = Query(default="eval"),
) -> dict[str, float]:
    _, public_alias = _normalize_task_ref(task_id)
    from graders.runtime import validator_grade_payload

    return validator_grade_payload(
        public_alias,
        episode_id=episode_id,
        seed=seed,
        split=split,
    )


@app.post("/grade/{task_id}")
def grade_task_post(
    task_id: str,
    req: GraderRequest | None = Body(default=None),
) -> dict[str, float]:
    return grade_task(
        task_id,
        episode_id=(req.episode_id if req is not None else None),
        seed=(req.seed if req is not None else 42),
        split=(req.split if req is not None else "eval"),
    )


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
