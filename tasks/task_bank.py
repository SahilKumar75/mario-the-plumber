"""Canonical root-level task registry for hackathon validators."""

from __future__ import annotations

from tasks.common import TaskDefinition, task_payload
from tasks.task1 import TASK as TASK_1
from tasks.task2 import TASK as TASK_2
from tasks.task3 import TASK as TASK_3

TASKS: dict[str, TaskDefinition] = {
    task.id: task
    for task in (TASK_1, TASK_2, TASK_3)
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
