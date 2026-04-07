"""Helpers for translating between public task aliases and internal task ids."""

from __future__ import annotations

from benchmark.catalog import TASK_NAMES


def public_task_id(task_id: int) -> str:
    """Return the public-facing task alias used by validator-style endpoints."""

    if task_id not in TASK_NAMES:
        raise ValueError(f"Unsupported task_id: {task_id}")
    return f"task_{task_id}"


def parse_task_id(task_ref: int | str | None) -> int:
    """Normalize numeric or alias task references to the internal integer id."""

    if task_ref is None:
        return 1
    if isinstance(task_ref, int):
        if task_ref in TASK_NAMES:
            return task_ref
        raise ValueError(f"Unsupported task_id: {task_ref}")

    normalized = str(task_ref).strip().lower().replace("-", "_")
    if normalized.isdigit():
        task_id = int(normalized)
    elif normalized.startswith("task_") and normalized[5:].isdigit():
        task_id = int(normalized[5:])
    else:
        raise ValueError(f"Unsupported task_id: {task_ref}")

    if task_id not in TASK_NAMES:
        raise ValueError(f"Unsupported task_id: {task_ref}")
    return task_id
