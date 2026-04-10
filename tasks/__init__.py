"""Root-level task registry exports for hackathon validators."""

from __future__ import annotations

from importlib import import_module

from tasks.definitions import (
    TASKS,
    TaskDefinition,
    build_task_definition,
    get_task,
    list_internal_task_ids,
    list_task_ids,
    list_tasks,
    task_payload,
    task_payloads,
    tasks_payload,
)

_COMPAT_MODULES = {
    "task1_alert_prioritization": "tasks.task1_alert_prioritization",
    "task2_threat_detection": "tasks.task2_threat_detection",
    "task3_incident_response": "tasks.task3_incident_response",
}


def __getattr__(name: str):
    if name in _COMPAT_MODULES:
        return import_module(_COMPAT_MODULES[name])
    if name == "task1":
        return import_module(_COMPAT_MODULES["task1_alert_prioritization"])
    if name == "task2":
        return import_module(_COMPAT_MODULES["task2_threat_detection"])
    if name == "task3":
        return import_module(_COMPAT_MODULES["task3_incident_response"])
    raise AttributeError(f"module 'tasks' has no attribute '{name}'")

__all__ = [
    "TASKS",
    "TaskDefinition",
    "build_task_definition",
    "get_task",
    "list_internal_task_ids",
    "list_task_ids",
    "list_tasks",
    "task_payload",
    "task_payloads",
    "tasks_payload",
    "task1_alert_prioritization",
    "task2_threat_detection",
    "task3_incident_response",
    "task1",
    "task2",
    "task3",
]
