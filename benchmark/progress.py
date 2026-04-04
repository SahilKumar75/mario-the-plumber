from __future__ import annotations

try:
    from .catalog import FORMAL_TASK_SPECS
    from .grading import calculation_mismatch_count, task5_rollup_consistency_score
    from .observation_support import dependency_alerts, missing_expected_columns
    from .runtime_state import commit_ready
    from .actions.validation import table_has_structural_issues
except ImportError:
    from benchmark.catalog import FORMAL_TASK_SPECS
    from benchmark.grading import calculation_mismatch_count, task5_rollup_consistency_score
    from benchmark.observation_support import dependency_alerts, missing_expected_columns
    from benchmark.runtime_state import commit_ready
    from benchmark.actions.validation import table_has_structural_issues


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
    if env._task_id == 3:
        return {
            "repair_customers": not table_has_structural_issues(env, "customers"),
            "repair_products": not table_has_structural_issues(env, "products"),
            "repair_orders": not table_has_structural_issues(env, "orders"),
            "restore_dependency_consistency": calculation_mismatch_count(
                env._tables["orders"], env._tables["products"]
            ) == 0,
            "commit_pipeline": bool(env._state.done and env._state.success),
        }
    if env._task_id == 4:
        return {
            "normalize_orders_stream": not table_has_structural_issues(env, "orders"),
            "scale_resources_if_needed": (
                env._state.resource_level >= env._state.required_resource_level
                if env._state.backlog_rows > 0
                else True
            ),
            "load_incremental_backlog": env._state.backlog_rows == 0 and env._state.pending_batches == 0,
            "refresh_daily_summary": (
                not bool(env._scenario_meta.get("downstream_stale", False))
                and env._state.freshness_lag_minutes == 0
            ),
            "commit_recovery": bool(env._state.done and env._state.success),
        }
    if env._task_id == 5:
        schema_ok = not table_has_structural_issues(env, "source_orders") and not table_has_structural_issues(
            env, "catalog"
        )
        no_aliases = not missing_expected_columns(env, "source_orders") and not missing_expected_columns(
            env, "catalog"
        )
        rollup_consistent = task5_rollup_consistency_score(
            env._tables["source_orders"],
            env._tables["catalog"],
            env._tables["hourly_rollup"],
        ) >= 0.9999
        return {
            "reconcile_schema_aliases": no_aliases,
            "repair_catalog_and_source_quality": schema_ok,
            "replay_late_batches": env._state.backlog_rows == 0 and env._state.pending_batches == 0,
            "refresh_temporal_rollup": (
                not table_has_structural_issues(env, "hourly_rollup")
                and not bool(env._scenario_meta.get("downstream_stale", False))
                and rollup_consistent
            ),
            "meet_freshness_sla": env._state.freshness_lag_minutes <= 30,
            "commit_temporal_pipeline": bool(env._state.done and env._state.success),
        }
    return {}


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
    if env._task_id == 3:
        return {
            **trace_summary,
            "customer_contract": "stable" if not table_has_structural_issues(env, "customers") else "repair_required",
            "product_contract": "stable" if not table_has_structural_issues(env, "products") else "repair_required",
            "order_dependency": "consistent" if not dependency_alerts(env) else "cascading_breakage",
        }
    if env._task_id == 4:
        return {
            **trace_summary,
            "incremental_backlog": "cleared" if env._state.backlog_rows == 0 else "pending_replay",
            "summary_state": "fresh" if not bool(env._scenario_meta.get("downstream_stale", False)) else "stale",
            "recovery_gate": "safe_to_commit" if commit_ready(env) else "recovery_incomplete",
        }
    if env._task_id == 5:
        rollup_consistency = task5_rollup_consistency_score(
            env._tables["source_orders"],
            env._tables["catalog"],
            env._tables["hourly_rollup"],
        )
        return {
            **trace_summary,
            "schema_alignment": "stable" if not missing_expected_columns(env, "source_orders") else "schema_drift_active",
            "temporal_backfill": "cleared" if env._state.backlog_rows == 0 else "late_batches_pending",
            "rollup_state": (
                "fresh"
                if not bool(env._scenario_meta.get("downstream_stale", False)) and rollup_consistency >= 0.9999
                else "rollup_stale"
            ),
            "recovery_gate": "safe_to_commit" if commit_ready(env) else "recovery_incomplete",
        }
    return {"single_table_state": "stable" if not env._recent_errors else "repair_required"}
