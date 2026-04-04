"""Shared policy utility functions."""

from __future__ import annotations

from typing import Any

try:
    from ...models import PipelineDoctorAction, PipelineDoctorObservation
except ImportError:
    from models import PipelineDoctorAction, PipelineDoctorObservation


def action_has_required_fields(action: PipelineDoctorAction) -> bool:
    if action.action_id in {3, 4, 5, 6, 7, 8, 9, 11, 12} and not action.target_column:
        return False
    if action.action_id == 12 and not action.new_name:
        return False
    if action.action_id == 13 and not action.column_order:
        return False
    return True


def first_schema_mismatch(
    observation: PipelineDoctorObservation,
) -> tuple[str, dict[str, Any]] | None:
    for column, info in observation.schema_report.items():
        return column, info
    return None


def alias_fix_action(
    observation: PipelineDoctorObservation,
) -> PipelineDoctorAction | None:
    for drifted_name, expected_name in observation.column_alias_hints.items():
        return PipelineDoctorAction(
            action_id=12,
            target_column=drifted_name,
            new_name=expected_name,
        )
    return None


def repair_action_for_mismatch(
    column: str,
    info: dict[str, Any],
) -> PipelineDoctorAction:
    expected = info.get("expected")
    actual = info.get("actual")
    if expected == "int64":
        if actual == "object":
            return PipelineDoctorAction(action_id=4, target_column=column)
        return PipelineDoctorAction(action_id=7, target_column=column)
    if expected == "float64":
        if actual == "object":
            return PipelineDoctorAction(action_id=3, target_column=column)
        return PipelineDoctorAction(action_id=8, target_column=column)
    if expected == "object":
        return PipelineDoctorAction(action_id=9, target_column=column)
    return PipelineDoctorAction(action_id=14)


def column_from_errors(error_text: str, keyword: str) -> str | None:
    for chunk in error_text.split("|"):
        chunk = chunk.strip()
        if keyword not in chunk or ":" not in chunk:
            continue
        return chunk.split(":", 1)[0].strip()
    return None


def table_needs_attention(observation: PipelineDoctorObservation) -> bool:
    return bool(
        observation.missing_rate > 0
        or observation.duplicate_rate > 0
        or observation.type_violations > 0
        or observation.outlier_count > 0
        or observation.format_issues > 0
        or (observation.stage in {"orders", "source_orders"} and observation.backlog_rows > 0)
        or (observation.stage in {"daily_summary", "hourly_rollup"} and observation.downstream_stale)
        or (observation.stage == "hourly_rollup" and observation.freshness_lag_minutes > 30)
    )


def has_calculation_mismatch(observation: PipelineDoctorObservation) -> bool:
    return any("calculation mismatch" in error.lower() for error in observation.recent_errors)


def only_calculation_mismatch(observation: PipelineDoctorObservation) -> bool:
    return bool(
        observation.stage == "orders"
        and observation.missing_rate == 0
        and observation.duplicate_rate == 0
        and observation.type_violations == 0
        and observation.outlier_count == 0
        and observation.format_issues == 0
        and has_calculation_mismatch(observation)
    )


def table_should_advance(task_id: int, env, observation: PipelineDoctorObservation) -> bool:
    if task_id == 3 and only_calculation_mismatch(observation):
        return not (
            observation.subgoal_progress.get("repair_customers", False)
            and observation.subgoal_progress.get("repair_products", False)
        )
    if task_id == 4:
        if env.state.active_table == "orders":
            return observation.backlog_rows == 0 and not table_needs_attention(observation)
        if env.state.active_table == "products":
            return not table_needs_attention(observation)
        return False
    if task_id == 5:
        if env.state.active_table == "source_orders":
            return observation.backlog_rows == 0 and not table_needs_attention(observation)
        if env.state.active_table == "catalog":
            return not table_needs_attention(observation)
        return False
    if not env.state.done and not table_needs_attention(observation):
        return True
    if env.state.active_table == "orders" and observation.current_score >= 0.9:
        return True
    return False


def next_table(current_table: str, task_id: int = 3) -> str | None:
    if task_id == 4:
        order = ["orders", "products", "daily_summary"]
    elif task_id == 5:
        order = ["source_orders", "catalog", "hourly_rollup"]
    else:
        order = ["orders", "customers", "products"]
    if current_table not in order:
        return None
    index = order.index(current_table)
    if index + 1 >= len(order):
        return None
    return order[index + 1]


def same_action(left: PipelineDoctorAction, right: PipelineDoctorAction) -> bool:
    return left.model_dump(exclude_none=True) == right.model_dump(exclude_none=True)
