"""Observation packaging helpers for Mario."""

from __future__ import annotations

from benchmark.catalog import SYNTHETIC_DATA_NOTES, TASK_CARDS, TASK_OBJECTIVE_WEIGHTS
from benchmark.evaluation import objective_breakdown
from benchmark.inspection import format_issue_count, outlier_count, schema_report
from benchmark.observation_support import (
    column_alias_hints,
    dependency_alerts,
    missing_expected_columns,
    orchestration_alerts,
    table_health,
    workload_pressure,
)
from benchmark.progress import (
    backlog_age_minutes,
    dependency_health_summary,
    drift_markers,
    recent_failure_counters,
    sla_severity,
    task_progress_bundle,
)
from benchmark.runtime_state import commit_ready, current_frame
from models import PipelineDoctorObservation


def _task_sensitive_context(
    env,
    *,
    subgoal_progress: dict[str, bool],
    subgoal_order: list[str],
    active_subgoal: str,
    reward_machine_state: str,
) -> dict[str, object]:
    if env._task_id >= 3:
        return {
            "recent_failure_counters": recent_failure_counters(env),
            "drift_markers": drift_markers(env),
            "dependency_health_summary": dependency_health_summary(env),
            "tradeoff_weights": dict(TASK_OBJECTIVE_WEIGHTS.get(env._task_id, {})),
            "subgoal_progress": subgoal_progress,
            "subgoal_order": subgoal_order,
            "active_subgoal": active_subgoal,
            "reward_machine_state": reward_machine_state,
            "adaptation_target": str(env._scenario_meta.get("adaptation_target", "")),
        }
    return {
        "recent_failure_counters": {},
        "drift_markers": [],
        "dependency_health_summary": {},
        "tradeoff_weights": {},
        "subgoal_progress": {},
        "subgoal_order": [],
        "active_subgoal": "",
        "reward_machine_state": "",
        "adaptation_target": "",
    }


def build_observation(
    env,
    *,
    reward: float,
    done: bool,
    action_result: str = "",
    metadata: dict[str, object] | None = None,
) -> PipelineDoctorObservation:
    current = current_frame(env)
    total_cells = max(len(current) * max(len(current.columns), 1), 1)
    missing_rate = float(current.isnull().sum().sum() / total_cells)
    duplicate_rate = float(current.duplicated().sum() / max(len(current), 1))
    schema_report_payload = schema_report(env)
    alias_hints = column_alias_hints(env)
    subgoal_progress, subgoal_order, active_subgoal, reward_machine_state = task_progress_bundle(env)
    context = _task_sensitive_context(
        env,
        subgoal_progress=subgoal_progress,
        subgoal_order=subgoal_order,
        active_subgoal=active_subgoal,
        reward_machine_state=reward_machine_state,
    )
    task_card = TASK_CARDS.get(env._task_id, {})
    format_issues = format_issue_count(env)
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
        format_issues=format_issues,
        schema_report=schema_report_payload,
        recent_errors=env._recent_errors[:5],
        current_score=env._state.current_score,
        steps_taken=env._state.step_count,
        repeated_action_streak=env._state.repeated_action_streak,
        repeated_action_tripwire=env._state.repeated_action_tripwire,
        stage=env._state.active_table,
        available_actions=list(range(20)),
        action_result=action_result,
        table_health=table_health(env),
        dependency_alerts=dependency_alerts(env),
        commit_ready=commit_ready(env),
        scenario_split=env._split,
        schema_drift_count=len(schema_report_payload) + format_issues,
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
        recent_failure_counters=context["recent_failure_counters"],
        drift_markers=context["drift_markers"],
        dependency_health_summary=context["dependency_health_summary"],
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
        synthetic_data_notes=list(env._scenario_meta.get("synthetic_data_notes", SYNTHETIC_DATA_NOTES)),
        reward_breakdown=dict(env._last_reward_breakdown),
        objective_breakdown=objective_breakdown(env),
        tradeoff_weights=context["tradeoff_weights"],
        subgoal_progress=context["subgoal_progress"],
        subgoal_order=context["subgoal_order"],
        active_subgoal=context["active_subgoal"],
        reward_machine_state=context["reward_machine_state"],
        adaptation_target=context["adaptation_target"],
        heldout_profile_family=bool(env._scenario_meta.get("heldout_profile_family", False)),
        reward=reward,
        done=done,
        metadata=metadata or {},
    )
