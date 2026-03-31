"""Heuristic policy rules for Mario baseline modes."""

from __future__ import annotations

try:
    from ...models import PipelineDoctorAction, PipelineDoctorObservation
except ImportError:
    from models import PipelineDoctorAction, PipelineDoctorObservation

try:
    from ..utils import (
        alias_fix_action,
        column_from_errors,
        first_schema_mismatch,
        repair_action_for_mismatch,
        table_needs_attention,
    )
except ImportError:
    from benchmark.policies.utils import (
        alias_fix_action,
        column_from_errors,
        first_schema_mismatch,
        repair_action_for_mismatch,
        table_needs_attention,
    )

FALLBACK_ACTION = PipelineDoctorAction(action_id=14)


def heuristic_action_for(
    task_id: int,
    observation: PipelineDoctorObservation,
) -> PipelineDoctorAction:
    if task_id == 3:
        return task3_heuristic_action(observation)
    if task_id == 4:
        return task4_heuristic_action(observation)
    if task_id == 5:
        return task5_heuristic_action(observation)

    error_text = " | ".join(observation.recent_errors).lower()
    mismatch = first_schema_mismatch(observation)
    alias_action = alias_fix_action(observation)

    if alias_action:
        return alias_action
    if observation.duplicate_rate > 0:
        return PipelineDoctorAction(action_id=10)

    null_column = column_from_errors(error_text, "null")
    if null_column:
        if "expected int64" in error_text or "expected float64" in error_text:
            return PipelineDoctorAction(action_id=4, target_column=null_column)
        return PipelineDoctorAction(action_id=5, target_column=null_column)

    if mismatch:
        column, info = mismatch
        return repair_action_for_mismatch(column, info)

    format_column = column_from_errors(error_text, "format mismatch")
    if format_column:
        return PipelineDoctorAction(action_id=9, target_column=format_column)

    if observation.outlier_count > 0:
        outlier_column = column_from_errors(error_text, "outlier")
        if outlier_column:
            return PipelineDoctorAction(action_id=11, target_column=outlier_column)

    if observation.commit_ready:
        return PipelineDoctorAction(action_id=15)

    return FALLBACK_ACTION


def task3_heuristic_action(
    observation: PipelineDoctorObservation,
) -> PipelineDoctorAction:
    error_text = " | ".join(observation.recent_errors).lower()
    mismatch = first_schema_mismatch(observation)
    alias_action = alias_fix_action(observation)
    if alias_action:
        return alias_action

    if observation.stage == "orders":
        if observation.duplicate_rate > 0:
            return PipelineDoctorAction(action_id=10)
        format_column = column_from_errors(error_text, "format mismatch")
        if format_column:
            return PipelineDoctorAction(action_id=9, target_column=format_column)
        if mismatch:
            column, info = mismatch
            return repair_action_for_mismatch(column, info)
        if only_calculation_mismatch(observation):
            return PipelineDoctorAction(action_id=0, target_column="customers")

    elif observation.stage == "customers":
        null_column = column_from_errors(error_text, "null")
        if null_column:
            return PipelineDoctorAction(action_id=4, target_column=null_column)
        format_column = column_from_errors(error_text, "format mismatch")
        if format_column:
            return PipelineDoctorAction(action_id=9, target_column=format_column)
        if mismatch:
            column, info = mismatch
            return repair_action_for_mismatch(column, info)
        if not table_needs_attention(observation):
            return PipelineDoctorAction(action_id=0, target_column="products")

    elif observation.stage == "products":
        if observation.duplicate_rate > 0:
            return PipelineDoctorAction(action_id=10)
        format_column = column_from_errors(error_text, "format mismatch")
        if format_column:
            return PipelineDoctorAction(action_id=9, target_column=format_column)
        if observation.outlier_count > 0:
            return PipelineDoctorAction(action_id=11, target_column="unit_price")
        if mismatch:
            column, info = mismatch
            return repair_action_for_mismatch(column, info)
        if not table_needs_attention(observation):
            return PipelineDoctorAction(action_id=15)
    return FALLBACK_ACTION


def task4_heuristic_action(
    observation: PipelineDoctorObservation,
) -> PipelineDoctorAction:
    error_text = " | ".join(observation.recent_errors).lower()
    mismatch = first_schema_mismatch(observation)
    alias_action = alias_fix_action(observation)
    if alias_action:
        return alias_action

    if observation.stage == "orders":
        if observation.backlog_rows > 0:
            if observation.resource_level < observation.required_resource_level:
                return PipelineDoctorAction(action_id=16)
            return PipelineDoctorAction(action_id=18)
        if observation.duplicate_rate > 0:
            return PipelineDoctorAction(action_id=10)
        format_column = column_from_errors(error_text, "format mismatch")
        if format_column:
            return PipelineDoctorAction(action_id=9, target_column=format_column)
        if mismatch:
            column, info = mismatch
            return repair_action_for_mismatch(column, info)
        if not table_needs_attention(observation):
            return PipelineDoctorAction(action_id=0, target_column="products")

    elif observation.stage == "products":
        if observation.duplicate_rate > 0:
            return PipelineDoctorAction(action_id=10)
        format_column = column_from_errors(error_text, "format mismatch")
        if format_column:
            return PipelineDoctorAction(action_id=9, target_column=format_column)
        if observation.outlier_count > 0:
            outlier_column = column_from_errors(error_text, "outlier") or "unit_price"
            return PipelineDoctorAction(action_id=11, target_column=outlier_column)
        if mismatch:
            column, info = mismatch
            return repair_action_for_mismatch(column, info)
        if not table_needs_attention(observation):
            return PipelineDoctorAction(action_id=0, target_column="daily_summary")

    elif observation.stage == "daily_summary":
        format_column = column_from_errors(error_text, "format mismatch")
        if format_column:
            return PipelineDoctorAction(action_id=9, target_column=format_column)
        if mismatch:
            column, info = mismatch
            return repair_action_for_mismatch(column, info)
        if observation.downstream_stale or observation.freshness_lag_minutes > 0:
            return PipelineDoctorAction(action_id=19)
        if observation.resource_level > observation.required_resource_level:
            return PipelineDoctorAction(action_id=17)
        if observation.commit_ready:
            return PipelineDoctorAction(action_id=15)

    if observation.commit_ready:
        return PipelineDoctorAction(action_id=15)
    return FALLBACK_ACTION


def task5_heuristic_action(
    observation: PipelineDoctorObservation,
) -> PipelineDoctorAction:
    error_text = " | ".join(observation.recent_errors).lower()
    mismatch = first_schema_mismatch(observation)
    alias_action = alias_fix_action(observation)
    if alias_action:
        return alias_action

    if observation.stage == "source_orders":
        if observation.backlog_rows > 0:
            if observation.resource_level < observation.required_resource_level:
                return PipelineDoctorAction(action_id=16)
            return PipelineDoctorAction(action_id=18)
        if observation.duplicate_rate > 0:
            return PipelineDoctorAction(action_id=10)
        format_column = column_from_errors(error_text, "format mismatch")
        if format_column:
            return PipelineDoctorAction(action_id=9, target_column=format_column)
        if mismatch:
            column, info = mismatch
            return repair_action_for_mismatch(column, info)
        if not table_needs_attention(observation):
            return PipelineDoctorAction(action_id=0, target_column="catalog")

    elif observation.stage == "catalog":
        if observation.duplicate_rate > 0:
            return PipelineDoctorAction(action_id=10)
        format_column = column_from_errors(error_text, "format mismatch")
        if format_column:
            return PipelineDoctorAction(action_id=9, target_column=format_column)
        if observation.outlier_count > 0:
            outlier_column = column_from_errors(error_text, "outlier") or "unit_price"
            return PipelineDoctorAction(action_id=11, target_column=outlier_column)
        if mismatch:
            column, info = mismatch
            return repair_action_for_mismatch(column, info)
        if not table_needs_attention(observation):
            return PipelineDoctorAction(action_id=0, target_column="hourly_rollup")

    elif observation.stage == "hourly_rollup":
        format_column = column_from_errors(error_text, "format mismatch")
        if format_column:
            return PipelineDoctorAction(action_id=9, target_column=format_column)
        if mismatch:
            column, info = mismatch
            return repair_action_for_mismatch(column, info)
        if observation.downstream_stale or observation.freshness_lag_minutes > 30:
            return PipelineDoctorAction(action_id=19)
        if observation.resource_level > observation.required_resource_level:
            return PipelineDoctorAction(action_id=17)
        if observation.commit_ready:
            return PipelineDoctorAction(action_id=15)

    if observation.commit_ready:
        return PipelineDoctorAction(action_id=15)
    return FALLBACK_ACTION


def only_calculation_mismatch(observation: PipelineDoctorObservation) -> bool:
    return bool(
        observation.stage == "orders"
        and observation.missing_rate == 0
        and observation.duplicate_rate == 0
        and observation.type_violations == 0
        and observation.outlier_count == 0
        and observation.format_issues == 0
        and any("calculation mismatch" in error.lower() for error in observation.recent_errors)
    )
