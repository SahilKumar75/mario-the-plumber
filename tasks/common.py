"""Shared task registry models for the validator-facing Mario task bank."""

from __future__ import annotations

from dataclasses import dataclass

from benchmark.catalog import MAX_STEPS, TASK_CARDS, TASK_DIFFICULTY, TASK_NAMES, TASK_THRESHOLDS
from benchmark.task_ids import parse_task_id, public_task_id


@dataclass(frozen=True, slots=True)
class TaskDefinition:
    """Canonical task metadata shared by the server, inference, and graders."""

    id: str
    internal_id: int
    name: str
    difficulty: str
    description: str
    max_steps: int
    success_threshold: float
    grade_endpoint: str


def build_task_definition(task_ref: int | str) -> TaskDefinition:
    """Build a public task definition from the benchmark catalog."""

    task_id = parse_task_id(task_ref)
    task_alias = public_task_id(task_id)
    description = str(
        TASK_CARDS[task_id].get(
            "incident_description",
            TASK_CARDS[task_id].get("objective", ""),
        )
    )
    return TaskDefinition(
        id=task_alias,
        internal_id=task_id,
        name=TASK_NAMES[task_id],
        difficulty=TASK_DIFFICULTY[task_id],
        description=description,
        max_steps=MAX_STEPS[task_id],
        success_threshold=TASK_THRESHOLDS[task_id],
        grade_endpoint=f"/grade/{task_alias}",
    )


def task_payload(task: TaskDefinition) -> dict[str, object]:
    """Return a stable validator-facing payload for one task."""

    return {
        "id": task.id,
        "name": task.name,
        "difficulty": task.difficulty,
        "description": task.description,
        "grader": {
            "type": "http",
            "endpoint": task.grade_endpoint,
        },
        "grade_endpoint": task.grade_endpoint,
    }
