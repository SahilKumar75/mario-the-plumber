"""Lightweight trained policy loader for baseline comparisons."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from models import PipelineDoctorAction, PipelineDoctorObservation

ARTIFACT_PATH = Path(__file__).with_name("trained_policy_v1.json")


def observation_signature(task_id: int, observation: PipelineDoctorObservation) -> str:
    """Build a compact, deterministic state signature for policy lookup."""

    signature = {
        "task_id": task_id,
        "stage": observation.stage,
        "missing_bin": min(int(observation.missing_rate * 10), 9),
        "duplicate_bin": min(int(observation.duplicate_rate * 10), 9),
        "type_bucket": min(observation.type_violations, 3),
        "format_bucket": min(observation.format_issues, 3),
        "outlier_bucket": min(observation.outlier_count, 3),
        "commit_ready": int(observation.commit_ready),
        "backlog_present": int(observation.backlog_rows > 0),
        "pending_batches": min(observation.pending_batches, 6),
        "resource_gap": max(observation.required_resource_level - observation.resource_level, 0),
        "downstream_stale": int(observation.downstream_stale),
        "dependency_alerts": int(bool(observation.dependency_alerts)),
        "active_subgoal": observation.active_subgoal or "",
    }
    return json.dumps(signature, sort_keys=True)


@lru_cache(maxsize=1)
def load_trained_policy() -> dict[str, Any]:
    """Load the trained policy artifact once per process."""

    if not ARTIFACT_PATH.exists():
        return {"version": "missing", "task_policies": {}}
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def trained_action_for(
    task_id: int,
    observation: PipelineDoctorObservation,
    *,
    fallback_action: PipelineDoctorAction,
) -> PipelineDoctorAction:
    """Return trained-policy action for the observation or a safe fallback."""

    policy = load_trained_policy()
    task_policies = policy.get("task_policies", {})
    task_mapping = task_policies.get(str(task_id), {})
    signature = observation_signature(task_id, observation)
    match = task_mapping.get(signature)

    if not isinstance(match, dict):
        return fallback_action

    action_payload = match.get("action")
    if not isinstance(action_payload, dict):
        return fallback_action

    try:
        return PipelineDoctorAction(**action_payload)
    except Exception:
        return fallback_action
