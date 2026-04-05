from __future__ import annotations

from benchmark.actions.validation import table_has_structural_issues
from benchmark.grading import task5_rollup_consistency_score, task5_temporal_closure_score
from benchmark.observation_support import missing_expected_columns
from benchmark.runtime_state import commit_ready


def subgoal_progress_map(env) -> dict[str, bool]:
    schema_ok = not table_has_structural_issues(env, "source_orders") and not table_has_structural_issues(env, "catalog")
    no_aliases = not missing_expected_columns(env, "source_orders") and not missing_expected_columns(env, "catalog")
    rollup_consistent = task5_rollup_consistency_score(
        env._tables["source_orders"],
        env._tables["catalog"],
        env._tables["hourly_rollup"],
    ) >= 0.9999
    temporal_closure = task5_temporal_closure_score(
        env._tables["source_orders"],
        env._tables["hourly_rollup"],
        env._scenario_meta.get("incident_manifest"),
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
        "meet_freshness_sla": env._state.freshness_lag_minutes <= 30 and temporal_closure,
        "commit_temporal_pipeline": bool(env._state.done and env._state.success),
    }


def dependency_health_summary(env, trace_summary: dict[str, str]) -> dict[str, str]:
    rollup_consistency = task5_rollup_consistency_score(
        env._tables["source_orders"],
        env._tables["catalog"],
        env._tables["hourly_rollup"],
    )
    temporal_closure = task5_temporal_closure_score(
        env._tables["source_orders"],
        env._tables["hourly_rollup"],
        env._scenario_meta.get("incident_manifest"),
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
        "temporal_closure": "closed" if temporal_closure >= 0.9999 else "replay_window_open",
        "recovery_gate": "safe_to_commit" if commit_ready(env) else "recovery_incomplete",
    }


def runtime_errors(env) -> list[str]:
    errors: list[str] = []
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
    return errors
