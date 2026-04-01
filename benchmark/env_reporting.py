"""Observation, scoring, and episode reporting helpers for Mario."""

from __future__ import annotations

from uuid import uuid4

try:
    from .catalog import FORMAL_TASK_SPECS, TASK_CARDS, TASK_OBJECTIVE_WEIGHTS, TASK_THRESHOLDS
    from .grading import (
        calculation_mismatch_count,
        duplicate_row_count,
        score_single_table,
        score_task3,
        score_task4,
        score_task5,
    )
    from .observation_support import (
        column_alias_hints,
        dependency_alerts,
        format_issue_details_for_frame,
        missing_expected_columns,
        orchestration_alerts,
        outlier_details_for_frame,
        table_health,
        workload_pressure,
    )
except ImportError:
    from benchmark.catalog import FORMAL_TASK_SPECS, TASK_CARDS, TASK_OBJECTIVE_WEIGHTS, TASK_THRESHOLDS
    from benchmark.grading import (
        calculation_mismatch_count,
        duplicate_row_count,
        score_single_table,
        score_task3,
        score_task4,
        score_task5,
    )
    from benchmark.observation_support import (
        column_alias_hints,
        dependency_alerts,
        format_issue_details_for_frame,
        missing_expected_columns,
        orchestration_alerts,
        outlier_details_for_frame,
        table_health,
        workload_pressure,
    )

try:
    from ..models import PipelineDoctorObservation
except ImportError:
    from models import PipelineDoctorObservation


def build_observation(
    env,
    *,
    reward: float,
    done: bool,
    action_result: str = "",
    metadata: dict[str, object] | None = None,
) -> PipelineDoctorObservation:
    current = env._current_frame()
    total_cells = max(len(current) * max(len(current.columns), 1), 1)
    missing_rate = float(current.isnull().sum().sum() / total_cells)
    duplicate_rate = float(duplicate_row_count(current) / max(len(current), 1))

    schema_report_payload = schema_report(env)
    alias_hints = column_alias_hints(env)
    subgoal_progress, subgoal_order, active_subgoal, reward_machine_state = task_progress_bundle(env)
    task_card = TASK_CARDS.get(env._task_id, {})
    return PipelineDoctorObservation(
        incident_type=str(task_card.get("incident_type", "")),
        incident_summary=str(task_card.get("incident_description", "")),
        diagnosis_signals=list(task_card.get("diagnosis_signals", [])),
        recovery_requirements=list(task_card.get("recovery_requirements", [])),
        unsafe_commit_conditions=list(task_card.get("unsafe_commit_conditions", [])),
        threshold_rationale=str(task_card.get("threshold_rationale", "")),
        missing_rate=round(missing_rate, 4),
        duplicate_rate=round(duplicate_rate, 4),
        type_violations=len(schema_report_payload),
        outlier_count=outlier_count(env),
        format_issues=format_issue_count(env),
        schema_report=schema_report_payload,
        recent_errors=env._recent_errors[:5],
        current_score=env._state.current_score,
        steps_taken=env._state.step_count,
        stage=env._state.active_table,
        available_actions=list(range(20)),
        action_result=action_result,
        table_health=table_health(env),
        dependency_alerts=dependency_alerts(env),
        commit_ready=env._commit_ready(),
        scenario_split=env._split,
        schema_drift_count=len(schema_report_payload) + format_issue_count(env),
        backlog_rows=env._state.backlog_rows,
        queue_backlog_age_minutes=backlog_age_minutes(env),
        freshness_lag_minutes=env._state.freshness_lag_minutes,
        sla_severity=sla_severity(env),
        resource_level=env._state.resource_level,
        required_resource_level=env._state.required_resource_level,
        workload_pressure=workload_pressure(env),
        pending_batches=env._state.pending_batches,
        downstream_stale=bool(env._scenario_meta.get("downstream_stale", False)),
        orchestration_alerts=orchestration_alerts(env),
        recent_failure_counters=recent_failure_counters(env),
        drift_markers=drift_markers(env),
        dependency_health_summary=dependency_health_summary(env),
        observed_columns=list(current.columns),
        missing_expected_columns=missing_expected_columns(env, env._state.active_table),
        column_alias_hints=alias_hints,
        scenario_profile=str(env._scenario_meta.get("scenario_profile", "baseline")),
        open_world_patterns=list(env._scenario_meta.get("open_world_patterns", [])),
        time_budget_remaining=max(0, env._state.max_steps - env._state.step_count),
        time_budget_ratio=round(
            max(0.0, (env._state.max_steps - env._state.step_count) / max(env._state.max_steps, 1)),
            4,
        ),
        truncated=env._state.truncated,
        done_reason=env._state.done_reason,
        synthetic_data_notes=list(env._scenario_meta.get("synthetic_data_notes", [])),
        reward_breakdown=dict(env._last_reward_breakdown),
        objective_breakdown=objective_breakdown(env),
        tradeoff_weights=dict(TASK_OBJECTIVE_WEIGHTS.get(env._task_id, {})),
        subgoal_progress=subgoal_progress,
        subgoal_order=subgoal_order,
        active_subgoal=active_subgoal,
        reward_machine_state=reward_machine_state,
        adaptation_target=str(env._scenario_meta.get("adaptation_target", "")),
        heldout_profile_family=bool(env._scenario_meta.get("heldout_profile_family", False)),
        reward=reward,
        done=done,
        metadata=metadata or {},
    )


def schema_report(env) -> dict[str, dict[str, str]]:
    return schema_report_for_table(env, env._state.active_table)


def schema_report_for_table(env, table_name: str) -> dict[str, dict[str, str]]:
    current = env._tables[table_name]
    expected = env._expected_types[table_name]
    report: dict[str, dict[str, str]] = {}
    for column in current.columns:
        actual = str(current[column].dtype)
        desired = expected.get(column)
        if desired and actual != desired:
            report[column] = {"expected": desired, "actual": actual}
    for column in expected:
        if column not in current.columns:
            report[column] = {"expected": expected[column], "actual": "missing"}
    return report


def outlier_details(env) -> dict[str, int]:
    return outlier_details_for_frame(env, env._current_frame())


def outlier_count(env) -> int:
    return sum(outlier_details(env).values())


def format_issue_details(env) -> dict[str, int]:
    return format_issue_details_for_frame(env, env._current_frame())


def format_issue_count(env) -> int:
    return sum(format_issue_details(env).values())


def refresh_errors(env) -> None:
    current = env._current_frame()
    errors: list[str] = []
    null_counts = current.isnull().sum()
    for column, count in null_counts.items():
        if int(count) > 0:
            errors.append(f"{column}: {int(count)} null values")
    duplicate_count = duplicate_row_count(current)
    if duplicate_count > 0:
        errors.append(f"{duplicate_count} duplicate rows detected")
    for column, count in outlier_details(env).items():
        errors.append(f"{column}: {count} outlier values")
    for column, count in format_issue_details(env).items():
        errors.append(f"{column}: {count} format mismatch values")
    for column, info in schema_report(env).items():
        errors.append(f"{column}: expected {info['expected']}, found {info['actual']}")
    if env._task_id == 3 and env._state.active_table == "orders":
        mismatch_count = calculation_mismatch_count(env._tables["orders"], env._tables["products"])
        if mismatch_count > 0:
            errors.append(f"total_price: {mismatch_count} rows have calculation mismatch")
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
        )
        return value
    value, _ = score_single_table(
        env._tables["single"],
        env._ground_truth["single"],
        env._expected_types["single"],
    )
    return value


def store_episode_summary(env) -> None:
    threshold = TASK_THRESHOLDS[env._task_id]
    success = bool(env._state.done and env._state.current_score >= threshold)
    env.EPISODE_SUMMARIES[env._state.episode_id or str(uuid4())] = {
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
            "repair_customers": not env._table_has_structural_issues("customers"),
            "repair_products": not env._table_has_structural_issues("products"),
            "repair_orders": not env._table_has_structural_issues("orders"),
            "restore_dependency_consistency": calculation_mismatch_count(
                env._tables["orders"], env._tables["products"]
            ) == 0,
            "commit_pipeline": bool(env._state.done and env._state.success),
        }
    if env._task_id == 4:
        return {
            "normalize_orders_stream": not env._table_has_structural_issues("orders"),
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
        schema_ok = not env._table_has_structural_issues("source_orders") and not env._table_has_structural_issues(
            "catalog"
        )
        no_aliases = not missing_expected_columns(env, "source_orders") and not missing_expected_columns(
            env, "catalog"
        )
        return {
            "reconcile_schema_aliases": no_aliases,
            "repair_catalog_and_source_quality": schema_ok,
            "replay_late_batches": env._state.backlog_rows == 0 and env._state.pending_batches == 0,
            "refresh_temporal_rollup": (
                not env._table_has_structural_issues("hourly_rollup")
                and not bool(env._scenario_meta.get("downstream_stale", False))
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
    if missing_expected_columns(env, env._state.active_table):
        markers.append("missing_expected_columns")
    if column_alias_hints(env):
        markers.append("column_alias_drift")
    if schema_report(env):
        markers.append("schema_validation_regression")
    return sorted(dict.fromkeys(str(marker) for marker in markers))


def dependency_health_summary(env) -> dict[str, str]:
    if env._task_id == 3:
        return {
            "customer_contract": "stable" if not env._table_has_structural_issues("customers") else "repair_required",
            "product_contract": "stable" if not env._table_has_structural_issues("products") else "repair_required",
            "order_dependency": (
                "consistent"
                if not dependency_alerts(env)
                else "cascading_breakage"
            ),
        }
    if env._task_id == 4:
        return {
            "incremental_backlog": "cleared" if env._state.backlog_rows == 0 else "pending_replay",
            "summary_state": "fresh" if not bool(env._scenario_meta.get("downstream_stale", False)) else "stale",
            "recovery_gate": "safe_to_commit" if env._commit_ready() else "recovery_incomplete",
        }
    if env._task_id == 5:
        return {
            "schema_alignment": "stable" if not missing_expected_columns(env, "source_orders") else "schema_drift_active",
            "temporal_backfill": "cleared" if env._state.backlog_rows == 0 else "late_batches_pending",
            "rollup_state": "fresh" if not bool(env._scenario_meta.get("downstream_stale", False)) else "rollup_stale",
        }
    return {"single_table_state": "stable" if not env._recent_errors else "repair_required"}
