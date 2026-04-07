"""Observation and diagnostic helpers for the Mario environment."""

from __future__ import annotations

import re

import pandas as pd

from benchmark.actions.transforms import is_datetime_like_column, normalize_string_value
from benchmark.actions.validation import table_has_structural_issues
from benchmark.grading import calculation_mismatch_count, score_single_table
from benchmark.runtime_state import current_frame


def missing_expected_columns(env, table_name: str) -> list[str]:
    current = env._tables[table_name]
    expected = env._expected_types[table_name]
    return [column for column in expected if column not in current.columns]


def column_alias_hints(env) -> dict[str, str]:
    current_columns = set(current_frame(env).columns)
    expected_columns = set(env._expected_types[env._state.active_table])
    aliases = {
        "signup_dt": "signup_date",
        "product_category": "category",
        "product_segment": "category",
        "sku_id": "product_id",
        "item_sku": "product_id",
        "business_date": "event_date",
        "observed_at": "event_ts",
        "replay_observed_at": "event_ts",
        "window_start": "hour_bucket",
        "bucket_window_utc_start": "hour_bucket",
        "gross_sales": "gross_revenue",
        "revenue_total": "gross_revenue",
        "replayed_revenue_value": "gross_revenue",
        "revenue_usd_value": "gross_revenue",
        "catalog_segment_name": "category",
    }
    hints: dict[str, str] = {}
    for drifted_name, expected_name in aliases.items():
        if drifted_name in current_columns and expected_name not in current_columns:
            hints[drifted_name] = expected_name
    if "event_time" in current_columns:
        if "event_date" in expected_columns and "event_date" not in current_columns:
            hints["event_time"] = "event_date"
        elif "event_ts" in expected_columns and "event_ts" not in current_columns:
            hints["event_time"] = "event_ts"
    return hints


def outlier_details_for_frame(env, current: pd.DataFrame) -> dict[str, int]:
    details: dict[str, int] = {}
    for column in current.columns:
        selected = current[column]
        if not isinstance(selected, pd.Series):
            continue
        numeric = pd.to_numeric(selected, errors="coerce")
        if numeric.isna().all():
            continue
        std = float(numeric.std(skipna=True))
        if pd.isna(std) or std == 0:
            continue
        mean = float(numeric.mean(skipna=True))
        outlier_count = int(((numeric - mean).abs() > 3 * std).fillna(False).sum())
        if outlier_count > 0:
            details[column] = outlier_count
    return details


def format_issue_details_for_frame(env, current: pd.DataFrame) -> dict[str, int]:
    details: dict[str, int] = {}
    for column in current.columns:
        series = current[column]
        if not isinstance(series, pd.Series):
            continue
        issue_count = 0
        column_name = column.lower()
        for value in series.dropna():
            text = str(value)
            normalized = normalize_string_value(env, value, column)
            if is_datetime_like_column(column_name):
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(text).strip()):
                    if column_name.endswith("_ts") or column_name.endswith("_time"):
                        if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", str(text).strip()):
                            continue
                    issue_count += 1
                    continue
            elif column_name in {"email", "category", "status"} and text != str(normalized):
                issue_count += 1
        if issue_count > 0:
            details[column] = issue_count
    return details


def dependency_alerts(env) -> list[str]:
    if env._task_id == 5:
        alerts: list[str] = []
        incident_manifest = env._scenario_meta.get("incident_manifest", {})
        affected_buckets = incident_manifest.get("affected_hour_buckets", [])
        expected_watermark = incident_manifest.get("expected_watermark_after_replay")
        if env._state.backlog_rows > 0:
            alerts.append("late source batches still need replay into source_orders")
        if bool(env._scenario_meta.get("downstream_stale", False)):
            alerts.append("hourly_rollup is stale relative to source_orders and catalog")
        if env._state.freshness_lag_minutes > 30:
            alerts.append("freshness SLA is still violated for the temporal pipeline")
        if env._state.resource_level < env._state.required_resource_level and env._state.backlog_rows > 0:
            alerts.append("resource level is too low for late-batch replay")
        if affected_buckets:
            alerts.append(f"affected rollup buckets remain open: {', '.join(str(value) for value in affected_buckets[:2])}")
        if expected_watermark:
            alerts.append(f"replay watermark must advance to {expected_watermark}")
        return alerts[:4]
    if env._task_id == 4:
        alerts: list[str] = []
        if env._state.backlog_rows > 0:
            alerts.append("latest incremental batch is still pending in the orders stream")
        if bool(env._scenario_meta.get("downstream_stale", False)):
            alerts.append("daily_summary is stale relative to upstream orders/products")
        if env._state.resource_level < env._state.required_resource_level and env._state.backlog_rows > 0:
            alerts.append("resource level is too low for backlog recovery")
        return alerts[:3]
    if env._task_id != 3:
        return []
    alerts = []
    mismatch_count = calculation_mismatch_count(env._tables["orders"], env._tables["products"])
    if mismatch_count > 0:
        alerts.append("orders.total_price depends on products.unit_price and is still inconsistent")
    if table_has_structural_issues(env, "products"):
        alerts.append("products still blocks a safe task 3 commit")
    if table_has_structural_issues(env, "orders"):
        alerts.append("orders still contains structural issues")
    return alerts[:3]


def table_health(env) -> dict[str, float]:
    if env._task_id == 5:
        tables = ("source_orders", "catalog", "hourly_rollup")
    elif env._task_id == 4:
        tables = ("orders", "products", "daily_summary")
    elif env._task_id == 3:
        tables = ("orders", "customers", "products")
    else:
        return {"single": round(env._state.current_score, 4)}
    return {
        table_name: round(
            score_single_table(
                env._tables[table_name],
                env._ground_truth[table_name],
                env._expected_types[table_name],
            )[0],
            4,
        )
        for table_name in tables
    }


def workload_pressure(env) -> float:
    base_pressure = float(env._scenario_meta.get("workload_pressure", 0.0))
    if env._task_id not in {4, 5}:
        return round(base_pressure, 4)
    backlog_bonus = min(env._state.backlog_rows / 10.0, 0.4 if env._task_id == 4 else 0.5)
    resource_relief = 0.15 * max(env._state.resource_level - 1, 0)
    pressure = max(0.0, min(1.0, base_pressure + backlog_bonus - resource_relief))
    return round(pressure, 4)


def orchestration_alerts(env) -> list[str]:
    if env._task_id == 5:
        alerts: list[str] = []
        incident_manifest = env._scenario_meta.get("incident_manifest", {})
        novelty_axes = incident_manifest.get("novelty_axes", [])
        if env._state.backlog_rows > 0 and env._state.resource_level < env._state.required_resource_level:
            alerts.append("scale resources before replaying the held-out temporal batches")
        if env._state.backlog_rows == 0 and bool(env._scenario_meta.get("downstream_stale", False)):
            alerts.append("refresh hourly_rollup after replay to close the temporal task")
        if env._state.freshness_lag_minutes > 30:
            alerts.append("bring freshness lag below the 30-minute SLA before committing")
        if novelty_axes:
            alerts.append(f"trace novelty axes: {', '.join(str(axis) for axis in novelty_axes[:2])}")
        return alerts[:3]
    if env._task_id != 4:
        return []
    alerts: list[str] = []
    if env._state.backlog_rows > 0 and env._state.resource_level < env._state.required_resource_level:
        alerts.append("scale resources up before prioritizing the delayed batch")
    if env._state.backlog_rows == 0 and bool(env._scenario_meta.get("downstream_stale", False)):
        alerts.append("refresh daily_summary before committing the recovery")
    if env._state.freshness_lag_minutes > 0:
        alerts.append(f"freshness lag is still {env._state.freshness_lag_minutes} minutes")
    return alerts[:3]
