from __future__ import annotations

from uuid import uuid4

from benchmark.catalog import TASK_THRESHOLDS
from benchmark.grading import score_single_table, score_task3, score_task4, score_task5


def score(env) -> float:
    if env._task_id == 3:
        value, _ = score_task3(env._tables, env._ground_truth, env._expected_types)
        return value
    if env._task_id == 4:
        value, _ = score_task4(
            env._tables,
            env._ground_truth,
            env._expected_types,
            backlog_rows=env._state.backlog_rows,
            freshness_lag_minutes=env._state.freshness_lag_minutes,
            resource_level=env._state.resource_level,
            required_resource_level=env._state.required_resource_level,
            downstream_stale=bool(env._scenario_meta.get("downstream_stale", False)),
        )
        return value
    if env._task_id == 5:
        value, _ = score_task5(
            env._tables,
            env._ground_truth,
            env._expected_types,
            backlog_rows=env._state.backlog_rows,
            freshness_lag_minutes=env._state.freshness_lag_minutes,
            resource_level=env._state.resource_level,
            required_resource_level=env._state.required_resource_level,
            downstream_stale=bool(env._scenario_meta.get("downstream_stale", False)),
            incident_manifest=env._scenario_meta.get("incident_manifest"),
        )
        return value
    value, _ = score_single_table(
        env._tables["single"],
        env._ground_truth["single"],
        env._expected_types["single"],
    )
    return value


def breakdown_payload(env) -> dict[str, object]:
    if env._task_id == 3:
        _, breakdown = score_task3(env._tables, env._ground_truth, env._expected_types)
        return breakdown
    if env._task_id == 4:
        _, breakdown = score_task4(
            env._tables,
            env._ground_truth,
            env._expected_types,
            backlog_rows=env._state.backlog_rows,
            freshness_lag_minutes=env._state.freshness_lag_minutes,
            resource_level=env._state.resource_level,
            required_resource_level=env._state.required_resource_level,
            downstream_stale=bool(env._scenario_meta.get("downstream_stale", False)),
        )
        return breakdown
    if env._task_id == 5:
        _, breakdown = score_task5(
            env._tables,
            env._ground_truth,
            env._expected_types,
            backlog_rows=env._state.backlog_rows,
            freshness_lag_minutes=env._state.freshness_lag_minutes,
            resource_level=env._state.resource_level,
            required_resource_level=env._state.required_resource_level,
            downstream_stale=bool(env._scenario_meta.get("downstream_stale", False)),
            incident_manifest=env._scenario_meta.get("incident_manifest"),
        )
        return breakdown
    _, breakdown = score_single_table(
        env._tables["single"],
        env._ground_truth["single"],
        env._expected_types["single"],
    )
    return breakdown


def objective_breakdown(env) -> dict[str, float]:
    breakdown = breakdown_payload(env)
    if env._task_id in {1, 2} and isinstance(breakdown, dict):
        return {
            key: round(float(value), 4)
            for key, value in breakdown.items()
            if isinstance(value, (int, float))
        }
    pipeline = breakdown.get("pipeline", {}) if isinstance(breakdown, dict) else {}
    return {
        key: round(float(value), 4)
        for key, value in pipeline.items()
        if isinstance(value, (int, float))
    }


def store_episode_summary(env) -> None:
    threshold = TASK_THRESHOLDS[env._task_id]
    success = bool(env._state.done and env._state.current_score >= threshold)
    env._episode_summaries[env._state.episode_id or str(uuid4())] = {
        "task_id": env._task_id,
        "episode_id": env._state.episode_id,
        "score": round(env._state.current_score, 4),
        "breakdown": breakdown_payload(env),
        "success": success,
        "steps_taken": env._state.step_count,
        "truncated": env._state.truncated,
        "done_reason": env._state.done_reason,
        "scenario_profile": env._state.scenario_profile,
    }
