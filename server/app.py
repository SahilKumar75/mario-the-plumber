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

try:
    from ..models import PipelineDoctorAction, PipelineDoctorObservation
    from ..inference import run_baseline
    from .data_generator import MAX_STEPS, TASK_DIFFICULTY, TASK_NAMES, TASK_THRESHOLDS
    from .pipeline_doctor_environment import EPISODE_SUMMARIES, PipelineDoctorEnvironment
except ImportError:
    from inference import run_baseline
    from models import PipelineDoctorAction, PipelineDoctorObservation
    from server.data_generator import MAX_STEPS, TASK_DIFFICULTY, TASK_NAMES, TASK_THRESHOLDS
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
)


@app.get("/health")
def health() -> dict[str, str]:
    """Lightweight health endpoint for container readiness checks."""

    return {"status": "ok"}


@app.get("/tasks")
def get_tasks() -> dict[str, object]:
    """Expose the benchmark task list and action schema."""

    return {
        "tasks": [
            {
                "task_id": task_id,
                "name": TASK_NAMES[task_id],
                "difficulty": TASK_DIFFICULTY[task_id],
                "success_threshold": TASK_THRESHOLDS[task_id],
                "max_steps": MAX_STEPS[task_id],
            }
            for task_id in (1, 2, 3)
        ],
        "action_schema": {
            "action_id": "int (0-15, required)",
            "target_column": (
                "str (optional; required for actions 3-9, 11, 12; optional for "
                "action 0 when switching tables in task 3)"
            ),
            "new_name": "str (optional, required for action 12 only)",
            "column_order": "list[str] (optional, required for action 13 only)",
        },
    }


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


@app.post("/baseline")
def baseline() -> dict[str, object]:
    """Run the local submission baseline."""

    return run_baseline()


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the development server directly."""

    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
