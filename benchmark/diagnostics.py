from __future__ import annotations

try:
    from .inspection import structural_mismatch_errors
    from .observation_support import dependency_alerts, orchestration_alerts
except ImportError:
    from benchmark.inspection import structural_mismatch_errors
    from benchmark.observation_support import dependency_alerts, orchestration_alerts


def refresh_errors(env) -> None:
    errors = structural_mismatch_errors(env)
    if env._task_id == 4:
        if env._state.backlog_rows > 0:
            errors.append(f"backlog: {env._state.backlog_rows} pending rows still need ingestion")
        if env._state.pending_batches > 0:
            errors.append(f"pending_batches: {env._state.pending_batches} incremental batch remains unprocessed")
        if bool(env._scenario_meta.get("downstream_stale", False)):
            errors.append("daily_summary: downstream aggregate is stale")
        if env._state.resource_level < env._state.required_resource_level and env._state.backlog_rows > 0:
            errors.append(
                f"resources: level {env._state.resource_level} below required {env._state.required_resource_level}"
            )
    if env._task_id == 5:
        if env._state.backlog_rows > 0:
            errors.append(f"late_batches: {env._state.backlog_rows} rows still await replay")
        if env._state.pending_batches > 0:
            errors.append(f"pending_batches: {env._state.pending_batches} temporal batches remain unreplayed")
        if bool(env._scenario_meta.get("downstream_stale", False)):
            errors.append("hourly_rollup: downstream aggregate is stale")
        if env._state.freshness_lag_minutes > 30:
            errors.append(f"freshness_sla: lag is {env._state.freshness_lag_minutes} minutes")
        if env._state.resource_level < env._state.required_resource_level and env._state.backlog_rows > 0:
            errors.append(
                f"resources: level {env._state.resource_level} below temporal requirement {env._state.required_resource_level}"
            )
    errors.extend(dependency_alerts(env))
    errors.extend(orchestration_alerts(env))
    env._recent_errors = errors[:6]
