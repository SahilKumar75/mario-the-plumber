# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI app for the Mario the Plumber OpenEnv environment."""

from __future__ import annotations

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
from inference import run_baseline
from models import PipelineDoctorAction, PipelineDoctorObservation
from server.benchmark_demo import build_benchmark_demo
from server.pipeline_doctor_environment import EPISODE_SUMMARIES, PipelineDoctorEnvironment


class GraderRequest(BaseModel):
    """Lookup payload for the /grader endpoint."""

    task_id: int
    episode_id: str


app = create_app(
    PipelineDoctorEnvironment,
    PipelineDoctorAction,
    PipelineDoctorObservation,
    env_name="mario_the_plumber",
    max_concurrent_envs=4,
    gradio_builder=build_benchmark_demo,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Lightweight health endpoint for container readiness checks."""

    return {"status": "healthy"}


@app.get("/tasks")
def get_tasks() -> dict[str, object]:
    """Expose the benchmark task list and action schema."""

    return tasks_payload()


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

    payload = EPISODE_SUMMARIES.get(request.episode_id)
    if payload is None:
        return {
            "task_id": request.task_id,
            "episode_id": request.episode_id,
            "score": 0.0,
            "breakdown": {},
            "success": False,
            "steps_taken": 0,
            "error": "episode_id not found",
        }
    return payload


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
