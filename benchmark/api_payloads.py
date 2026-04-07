"""Shared HTTP payload builders for the Mario benchmark API."""

from __future__ import annotations

import os

from benchmark.catalog import (
    MAX_STEPS,
    TASK_DIFFICULTY,
    TASK_CARDS,
    TASK_NAMES,
    TASK_THRESHOLDS,
    benchmark_metadata,
)
from benchmark.runtime import (
    adaptation_payload,
    benchmark_profiles_payload,
    benchmark_runs_payload,
    benchmark_tasks_payload,
    runtime_summary,
)
from benchmark.task_ids import public_task_id


def _base_url() -> str:
    """Return the optional public base URL without a trailing slash."""

    return os.getenv("MARIO_PUBLIC_BASE_URL", "").strip().rstrip("/")


def _grader_url() -> str:
    """Return the compatibility grader URL used for episode-summary lookup."""

    base_url = _base_url()
    if not base_url:
        return "/grader"
    return f"{base_url}/grader"


def _grade_url(task_id: int) -> str:
    """Return the task-local grade endpoint used by external validators."""

    task_alias = public_task_id(task_id)
    base_url = _base_url()
    if not base_url:
        return f"/grade/{task_alias}"
    return f"{base_url}/grade/{task_alias}"


def _task_payload(task_id: int) -> dict[str, object]:
    episode_grader = {
        "type": "http",
        "url": _grader_url(),
        "method": "POST",
        "content_type": "application/json",
        "payload_schema": {
            "task_id": task_id,
            "episode_id": "<episode_id>",
        },
    }
    grader = {
        "type": "http",
        "url": _grade_url(task_id),
        "method": "GET",
    }
    return {
        "id": public_task_id(task_id),
        "task_id": public_task_id(task_id),
        "internal_task_id": task_id,
        "name": TASK_NAMES[task_id],
        "difficulty": TASK_DIFFICULTY[task_id],
        "description": str(TASK_CARDS[task_id].get("incident_description", TASK_CARDS[task_id].get("objective", ""))),
        "success_threshold": TASK_THRESHOLDS[task_id],
        "max_steps": MAX_STEPS[task_id],
        "task_card": TASK_CARDS[task_id],
        # Some validators look only for a boolean `grader` flag and a `/grade/{task_id}`
        # endpoint, while others inspect richer endpoint metadata.
        "grader": True,
        "grader_enabled": True,
        "grade_endpoint": grader["url"],
        "graders": [grader],
        "grader_url": grader["url"],
        "grader_method": grader["method"],
        "grader_config": grader,
        "episode_grader": episode_grader,
    }


def _public_task_payload(task_id: int) -> dict[str, object]:
    return {
        "id": public_task_id(task_id),
        "description": str(
            TASK_CARDS[task_id].get(
                "incident_description",
                TASK_CARDS[task_id].get("objective", ""),
            )
        ),
        "max_steps": MAX_STEPS[task_id],
        "difficulty": TASK_DIFFICULTY[task_id],
        "grader": True,
    }


def public_tasks_payload() -> dict[str, object]:
    return {"tasks": [_public_task_payload(task_id) for task_id in sorted(TASK_NAMES)]}


def tasks_payload() -> dict[str, object]:
    return {
        "tasks": [_task_payload(task_id) for task_id in sorted(TASK_NAMES)],
        "action_schema": {
            "action_id": "int (0-19, required)",
            "target_column": (
                "str (optional; required for actions 3-9, 11, 12; optional for "
                "action 0 when switching tables in task 3, task 4, or task 5)"
            ),
            "new_name": "str (optional, required for action 12 only)",
            "column_order": "list[str] (optional, required for action 13 only)",
            "time_budget": "episodes end with truncation when max_steps is exhausted",
            "orchestration_actions": {
                "16": "scale_resources_up",
                "17": "scale_resources_down",
                "18": "prioritize_incremental_batch",
                "19": "refresh_downstream_summary",
            },
            "incident_signals": [
                "incident_type",
                "incident_summary",
                "diagnosis_signals",
                "recovery_requirements",
                "unsafe_commit_conditions",
                "queue_backlog_age_minutes",
                "sla_severity",
                "recent_failure_counters",
                "drift_markers",
                "dependency_health_summary",
            ],
            "reward_machine_signals": [
                "reward_breakdown",
                "objective_breakdown",
                "tradeoff_weights",
                "subgoal_progress",
                "reward_machine_state",
            ],
        },
    }


def benchmark_metadata_payload() -> dict[str, object]:
    return {
        **benchmark_metadata(),
        **runtime_summary(),
    }


__all__ = [
    "adaptation_payload",
    "benchmark_metadata_payload",
    "benchmark_profiles_payload",
    "benchmark_runs_payload",
    "benchmark_tasks_payload",
    "public_tasks_payload",
    "tasks_payload",
]
