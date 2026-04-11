"""Helpers for translating between public task aliases and internal task ids."""

from __future__ import annotations

from benchmark.catalog import TASK_NAMES


COMPAT_TASK_ALIASES: dict[str, int] = {
    "alert_prioritization": 1,
    "threat_detection": 2,
    "incident_response": 3,
}


def public_task_id(task_id: int) -> str:
    """Return the public-facing task alias used by validator-style endpoints."""

    if task_id not in TASK_NAMES:
        raise ValueError(f"Unsupported task_id: {task_id}")
    return f"task_{task_id}"


def list_compat_task_ids() -> list[str]:
    """Return supported compatibility task aliases in stable order."""

    return list(COMPAT_TASK_ALIASES)


def parse_task_id(task_ref: int | str | None) -> int:
    """Normalize numeric, task_N, or compatibility aliases to the internal id."""

    if task_ref is None:
        return 1
    if isinstance(task_ref, int):
        if task_ref in TASK_NAMES:
            return task_ref
        raise ValueError(f"Unsupported task_id: {task_ref}")

    normalized = str(task_ref).strip().lower().replace("-", "_")
    if normalized in COMPAT_TASK_ALIASES:
        task_id = COMPAT_TASK_ALIASES[normalized]
    elif normalized.isdigit():
        task_id = int(normalized)
    elif normalized.startswith("task_") and normalized[5:].isdigit():
        task_id = int(normalized[5:])
    else:
        raise ValueError(f"Unsupported task_id: {task_ref}")

    if task_id not in TASK_NAMES:
        raise ValueError(f"Unsupported task_id: {task_ref}")
    return task_id
