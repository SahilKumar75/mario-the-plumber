from __future__ import annotations

try:
    from ..runtime_state import commit_ready
    from ..actions.validation import table_has_structural_issues
except ImportError:
    from benchmark.runtime_state import commit_ready
    from benchmark.actions.validation import table_has_structural_issues


def subgoal_progress_map(env) -> dict[str, bool]:
    return {
        "normalize_orders_stream": not table_has_structural_issues(env, "orders"),
        "scale_resources_if_needed": (
            env._state.resource_level >= env._state.required_resource_level if env._state.backlog_rows > 0 else True
        ),
        "load_incremental_backlog": env._state.backlog_rows == 0 and env._state.pending_batches == 0,
        "refresh_daily_summary": (
            not bool(env._scenario_meta.get("downstream_stale", False)) and env._state.freshness_lag_minutes == 0
        ),
        "commit_recovery": bool(env._state.done and env._state.success),
    }


def dependency_health_summary(env, trace_summary: dict[str, str]) -> dict[str, str]:
    return {
        **trace_summary,
        "incremental_backlog": "cleared" if env._state.backlog_rows == 0 else "pending_replay",
        "summary_state": "fresh" if not bool(env._scenario_meta.get("downstream_stale", False)) else "stale",
        "recovery_gate": "safe_to_commit" if commit_ready(env) else "recovery_incomplete",
    }


def runtime_errors(env) -> list[str]:
    errors: list[str] = []
    if env._state.backlog_rows > 0:
        errors.append(f"backlog: {env._state.backlog_rows} pending rows still need ingestion")
    if env._state.pending_batches > 0:
        errors.append(f"pending_batches: {env._state.pending_batches} incremental batch remains unprocessed")
    if bool(env._scenario_meta.get("downstream_stale", False)):
        errors.append("daily_summary: downstream aggregate is stale")
    if env._state.resource_level < env._state.required_resource_level and env._state.backlog_rows > 0:
        errors.append(f"resources: level {env._state.resource_level} below required {env._state.required_resource_level}")
    return errors
