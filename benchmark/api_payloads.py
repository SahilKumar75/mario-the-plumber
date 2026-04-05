"""Shared HTTP payload builders for the Mario benchmark API."""

from __future__ import annotations

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


def tasks_payload() -> dict[str, object]:
    return {
        "tasks": [
            {
                "task_id": task_id,
                "name": TASK_NAMES[task_id],
                "difficulty": TASK_DIFFICULTY[task_id],
                "success_threshold": TASK_THRESHOLDS[task_id],
                "max_steps": MAX_STEPS[task_id],
                "task_card": TASK_CARDS[task_id],
            }
            for task_id in sorted(TASK_NAMES)
        ],
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
    "tasks_payload",
]
