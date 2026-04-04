from __future__ import annotations

try:
    from .catalog import FORMAL_TASK_SPECS
    from .observation_support import missing_expected_columns
    from .task_runtime.dispatch import dependency_health_summary as runtime_dependency_health_summary
    from .task_runtime.dispatch import subgoal_progress_map as runtime_subgoal_progress_map
except ImportError:
    from benchmark.catalog import FORMAL_TASK_SPECS
    from benchmark.observation_support import missing_expected_columns
    from benchmark.task_runtime.dispatch import dependency_health_summary as runtime_dependency_health_summary
    from benchmark.task_runtime.dispatch import subgoal_progress_map as runtime_subgoal_progress_map


def task_progress_bundle(env) -> tuple[dict[str, bool], list[str], str, str]:
    subgoal_progress = subgoal_progress_map(env)
    order = list(FORMAL_TASK_SPECS.get(env._task_id, {}).get("reward_machine_order", []))
    active_subgoal = next((name for name in order if not subgoal_progress.get(name, False)), "")
    completed = sum(1 for name in order if subgoal_progress.get(name, False))
    reward_machine_state = f"s{completed}/{len(order)}:{active_subgoal or 'terminal'}" if order else ""
    return subgoal_progress, order, active_subgoal, reward_machine_state


def update_task_progress_state(env) -> None:
    _, _, active_subgoal, reward_machine_state = task_progress_bundle(env)
    env._state.active_subgoal = active_subgoal
    env._state.reward_machine_state = reward_machine_state
    env._state.queue_backlog_age_minutes = backlog_age_minutes(env)
    env._state.sla_severity = sla_severity(env)


def subgoal_progress_map(env) -> dict[str, bool]:
    return runtime_subgoal_progress_map(env)


def backlog_age_minutes(env) -> int:
    if env._state.backlog_rows <= 0:
        return 0
    return int(env._scenario_meta.get("queue_backlog_age_minutes", env._state.queue_backlog_age_minutes))


def sla_severity(env) -> str:
    lag = int(env._state.freshness_lag_minutes)
    backlog_age = backlog_age_minutes(env)
    if lag <= 0 and backlog_age <= 0:
        return "none"
    if lag >= 180 or backlog_age >= 300:
        return "critical"
    if lag >= 90 or backlog_age >= 180:
        return "high"
    if lag > 0 or backlog_age > 0:
        return "elevated"
    return "none"


def recent_failure_counters(env) -> dict[str, int]:
    payload = env._scenario_meta.get("recent_failure_counters", {})
    if isinstance(payload, dict):
        return {str(key): int(value) for key, value in payload.items()}
    return {}


def drift_markers(env) -> list[str]:
    markers = list(env._scenario_meta.get("open_world_patterns", []))
    markers.extend(env._scenario_meta.get("trace_drift_markers", []))
    if missing_expected_columns(env, env._state.active_table):
        markers.append("missing_expected_columns")
    if env._task_id == 3 and "product_category" in env._tables["products"].columns:
        markers.append("column_alias_drift")
    if env._task_id == 4 and "event_time" in env._tables["orders"].columns:
        markers.append("column_alias_drift")
    if env._task_id == 5 and (
        "observed_at" in env._tables["source_orders"].columns or "window_start" in env._tables["hourly_rollup"].columns
    ):
        markers.append("column_alias_drift")
    return sorted(dict.fromkeys(str(marker) for marker in markers))


def dependency_health_summary(env) -> dict[str, str]:
    trace_payload = env._scenario_meta.get("trace_dependency_health", {})
    if isinstance(trace_payload, dict):
        trace_summary = {str(key): str(value) for key, value in trace_payload.items()}
    else:
        trace_summary = {}
    return runtime_dependency_health_summary(env, trace_summary)
