"""Canonical single-file root task definitions for hackathon validators."""

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
        "description": task.description,
        "grader": True,
        "grade_endpoint": task.grade_endpoint,
        "difficulty": task.difficulty,
    }


TASKS: dict[str, TaskDefinition] = {
    task.id: task
    for task in (
        build_task_definition(1),
        build_task_definition(2),
        build_task_definition(3),
    )
}


def list_task_ids() -> list[str]:
    return list(TASKS)


def list_internal_task_ids() -> list[int]:
    return [task.internal_id for task in TASKS.values()]


def list_tasks() -> list[TaskDefinition]:
    return [TASKS[task_id] for task_id in list_task_ids()]


def get_task(task_id: str) -> TaskDefinition:
    if task_id not in TASKS:
        raise KeyError(f"Unknown task_id: {task_id}")
    return TASKS[task_id]


def task_payloads() -> list[dict[str, object]]:
    return [
        {
            **task_payload(task),
            "id": task.id,
            "task_id": task.id,
            "name": task.name,
        }
        for task in list_tasks()
    ]


def tasks_payload() -> dict[str, object]:
    payloads = {
        task.id: task_payload(task)
        for task in list_tasks()
    }
    payloads["tasks"] = task_payloads()
    return payloads
