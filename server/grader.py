# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic scoring and reward helpers for PipelineDoctor."""

from __future__ import annotations

import pandas as pd


def duplicate_row_count(frame: pd.DataFrame) -> int:
    """Count duplicates using primary-key-like columns when available."""

    key_column = _primary_key_column(frame)
    if key_column and key_column in frame.columns:
        return int(frame.duplicated(subset=[key_column]).sum())
    return int(frame.duplicated().sum())


def score_single_table(
    fixed_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    expected_types: dict[str, str],
) -> tuple[float, dict[str, float]]:
    """Score one table against the provided ground truth."""

    total_cells = max(len(fixed_df) * max(len(fixed_df.columns), 1), 1)
    completeness = 1.0 - (fixed_df.isnull().sum().sum() / total_cells)

    validity_matches = sum(
        1
        for column in fixed_df.columns
        if str(fixed_df[column].dtype) == expected_types.get(column, "")
    )
    validity = validity_matches / max(len(fixed_df.columns), 1)

    consistency = 1.0
    if len(fixed_df) > 0:
        consistency = 1.0 - (duplicate_row_count(fixed_df) / len(fixed_df))

    accuracy = _accuracy(fixed_df, ground_truth_df)
    score = round(
        (0.20 * completeness)
        + (0.20 * validity)
        + (0.30 * consistency)
        + (0.30 * accuracy),
        4,
    )

    return score, {
        "completeness": round(completeness, 4),
        "validity": round(validity, 4),
        "consistency": round(consistency, 4),
        "accuracy": round(accuracy, 4),
    }


def score_task3(
    fixed_tables: dict[str, pd.DataFrame],
    ground_truth_tables: dict[str, pd.DataFrame],
    expected_types: dict[str, dict[str, str]],
) -> tuple[float, dict[str, dict[str, float]]]:
    """Weighted multi-table score for task 3."""

    orders_score, orders_breakdown = score_single_table(
        fixed_tables["orders"],
        ground_truth_tables["orders"],
        expected_types["orders"],
    )
    customers_score, customers_breakdown = score_single_table(
        fixed_tables["customers"],
        ground_truth_tables["customers"],
        expected_types["customers"],
    )
    products_score, products_breakdown = score_single_table(
        fixed_tables["products"],
        ground_truth_tables["products"],
        expected_types["products"],
    )
    base_score = (
        (0.50 * orders_score) + (0.30 * customers_score) + (0.20 * products_score)
    )
    dependency_score = task3_dependency_score(
        fixed_tables["orders"],
        fixed_tables["products"],
    )
    score = round(base_score * (0.30 + (0.70 * dependency_score)), 4)
    return score, {
        "orders": orders_breakdown,
        "customers": customers_breakdown,
        "products": products_breakdown,
        "pipeline": {"dependency_consistency": round(dependency_score, 4)},
    }


def score_task4(
    fixed_tables: dict[str, pd.DataFrame],
    ground_truth_tables: dict[str, pd.DataFrame],
    expected_types: dict[str, dict[str, str]],
    *,
    backlog_rows: int,
    freshness_lag_minutes: int,
    resource_level: int,
    required_resource_level: int,
    downstream_stale: bool,
) -> tuple[float, dict[str, dict[str, float]]]:
    """Weighted score for incremental ETL recovery with orchestration signals."""

    orders_score, orders_breakdown = score_single_table(
        fixed_tables["orders"],
        ground_truth_tables["orders"],
        expected_types["orders"],
    )
    products_score, products_breakdown = score_single_table(
        fixed_tables["products"],
        ground_truth_tables["products"],
        expected_types["products"],
    )
    summary_score, summary_breakdown = score_single_table(
        fixed_tables["daily_summary"],
        ground_truth_tables["daily_summary"],
        expected_types["daily_summary"],
    )

    base_score = (
        (0.40 * orders_score)
        + (0.20 * products_score)
        + (0.20 * summary_score)
    )
    batch_score = task4_batch_completeness_score(
        fixed_tables["orders"], ground_truth_tables["orders"], backlog_rows
    )
    freshness_score = max(0.0, 1.0 - (freshness_lag_minutes / 180.0))
    summary_consistency = task4_summary_consistency_score(
        fixed_tables["orders"],
        fixed_tables["products"],
        fixed_tables["daily_summary"],
    )
    if backlog_rows > 0 and resource_level < required_resource_level:
        resource_efficiency = 0.2
    else:
        overscale = max(resource_level - required_resource_level, 0)
        resource_efficiency = max(0.6, 1.0 - (0.15 * overscale))
    if downstream_stale:
        freshness_score *= 0.5

    orchestration_score = (
        0.15
        + (0.35 * batch_score)
        + (0.25 * freshness_score)
        + (0.15 * summary_consistency)
        + (0.10 * resource_efficiency)
    )
    score = round(base_score * orchestration_score, 4)
    return score, {
        "orders": orders_breakdown,
        "products": products_breakdown,
        "daily_summary": summary_breakdown,
        "pipeline": {
            "batch_completeness": round(batch_score, 4),
            "freshness": round(freshness_score, 4),
            "summary_consistency": round(summary_consistency, 4),
            "resource_efficiency": round(resource_efficiency, 4),
        },
    }


def score_task5(
    fixed_tables: dict[str, pd.DataFrame],
    ground_truth_tables: dict[str, pd.DataFrame],
    expected_types: dict[str, dict[str, str]],
    *,
    backlog_rows: int,
    freshness_lag_minutes: int,
    resource_level: int,
    required_resource_level: int,
    downstream_stale: bool,
) -> tuple[float, dict[str, dict[str, float]]]:
    """Temporal ETL score with explicit multi-objective structure."""

    source_score, source_breakdown = score_single_table(
        fixed_tables["source_orders"],
        ground_truth_tables["source_orders"],
        expected_types["source_orders"],
    )
    catalog_score, catalog_breakdown = score_single_table(
        fixed_tables["catalog"],
        ground_truth_tables["catalog"],
        expected_types["catalog"],
    )
    rollup_score, rollup_breakdown = score_single_table(
        fixed_tables["hourly_rollup"],
        ground_truth_tables["hourly_rollup"],
        expected_types["hourly_rollup"],
    )

    schema_alignment = 1.0 if source_breakdown["validity"] >= 1.0 and catalog_breakdown["validity"] >= 1.0 else max(
        0.0, (source_breakdown["validity"] + catalog_breakdown["validity"]) / 2.0
    )
    temporal_backfill = task4_batch_completeness_score(
        fixed_tables["source_orders"],
        ground_truth_tables["source_orders"],
        backlog_rows,
    )
    rollup_consistency = task5_rollup_consistency_score(
        fixed_tables["source_orders"],
        fixed_tables["catalog"],
        fixed_tables["hourly_rollup"],
    )
    freshness = max(0.0, 1.0 - (freshness_lag_minutes / 240.0))
    if downstream_stale:
        freshness *= 0.5

    if backlog_rows > 0 and resource_level < required_resource_level:
        resource_efficiency = 0.2
    else:
        overscale = max(resource_level - required_resource_level, 0)
        resource_efficiency = max(0.65, 1.0 - (0.10 * overscale))

    data_quality = (0.45 * source_score) + (0.25 * catalog_score) + (0.30 * rollup_score)
    score = round(
        (0.20 * schema_alignment)
        + (0.20 * temporal_backfill)
        + (0.20 * rollup_consistency)
        + (0.15 * freshness)
        + (0.10 * resource_efficiency)
        + (0.15 * data_quality),
        4,
    )
    return score, {
        "source_orders": source_breakdown,
        "catalog": catalog_breakdown,
        "hourly_rollup": rollup_breakdown,
        "pipeline": {
            "schema_alignment": round(schema_alignment, 4),
            "temporal_backfill": round(temporal_backfill, 4),
            "rollup_consistency": round(rollup_consistency, 4),
            "freshness": round(freshness, 4),
            "resource_efficiency": round(resource_efficiency, 4),
            "data_quality": round(data_quality, 4),
        },
    }


def compute_reward(
    score_before: float,
    score_after: float,
    *,
    action_valid: bool,
    done: bool,
    success: bool,
) -> float:
    """Reward shaping used by the environment."""

    reward = 0.5 * (score_after - score_before)
    reward -= 0.001
    if not action_valid:
        reward -= 0.1
    if done and success:
        reward += 1.0
    if done and not success:
        reward -= 0.5
    return round(reward, 4)


def compute_reward_breakdown(
    score_before: float,
    score_after: float,
    *,
    action_valid: bool,
    done: bool,
    success: bool,
) -> dict[str, float]:
    """Expose a structured reward explanation alongside the scalar reward."""

    delta = round(0.5 * (score_after - score_before), 4)
    breakdown = {
        "score_delta": delta,
        "step_cost": -0.001,
        "invalid_action_penalty": -0.1 if not action_valid else 0.0,
        "success_bonus": 1.0 if done and success else 0.0,
        "failure_penalty": -0.5 if done and not success else 0.0,
    }
    breakdown["total"] = round(sum(breakdown.values()), 4)
    return breakdown


def calculation_mismatch_count(
    orders_df: pd.DataFrame,
    products_df: pd.DataFrame,
) -> int:
    """Count wrong derived values for task 3's total_price column."""

    if "total_price" not in orders_df.columns or "unit_price" not in products_df.columns:
        return 0

    orders = orders_df.copy()
    products = products_df.copy()
    if "product_id" not in orders.columns or "product_id" not in products.columns:
        return len(orders_df)

    orders["product_id"] = pd.to_numeric(orders["product_id"], errors="coerce")
    products["product_id"] = pd.to_numeric(products["product_id"], errors="coerce")
    merged = orders.merge(
        products[["product_id", "unit_price"]],
        on="product_id",
        how="left",
        suffixes=("", "_product"),
    )
    quantity = pd.to_numeric(merged["quantity"], errors="coerce")
    current_total = pd.to_numeric(merged["total_price"], errors="coerce")
    expected_total = quantity * pd.to_numeric(merged["unit_price"], errors="coerce")
    return int((current_total.round(4) != expected_total.round(4)).fillna(True).sum())


def task3_dependency_score(
    orders_df: pd.DataFrame,
    products_df: pd.DataFrame,
) -> float:
    """Score whether cross-table derived values are internally consistent."""

    total_rows = max(len(orders_df), 1)
    mismatch_count = calculation_mismatch_count(orders_df, products_df)
    return max(0.0, 1.0 - (mismatch_count / total_rows))


def task4_batch_completeness_score(
    orders_df: pd.DataFrame,
    truth_orders_df: pd.DataFrame,
    backlog_rows: int,
) -> float:
    """Measure whether the latest incremental batch has been ingested."""

    if backlog_rows == 0 and len(orders_df) == len(truth_orders_df):
        return 1.0
    if "order_id" not in orders_df.columns or "order_id" not in truth_orders_df.columns:
        return 0.0
    fixed_ids = set(pd.to_numeric(orders_df["order_id"], errors="coerce").dropna().astype(int))
    truth_ids = set(pd.to_numeric(truth_orders_df["order_id"], errors="coerce").dropna().astype(int))
    if not truth_ids:
        return 1.0
    return len(fixed_ids.intersection(truth_ids)) / len(truth_ids)


def task4_summary_consistency_score(
    orders_df: pd.DataFrame,
    products_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> float:
    """Score whether downstream daily summary matches current upstream tables."""

    if not {"product_id", "quantity", "event_ts"}.issubset(orders_df.columns):
        return 0.0
    if not {"product_id", "unit_price"}.issubset(products_df.columns):
        return 0.0
    if not {"event_date", "order_count", "total_revenue"}.issubset(summary_df.columns):
        return 0.0

    orders = orders_df.copy()
    products = products_df.copy()
    orders["product_id"] = pd.to_numeric(orders["product_id"], errors="coerce")
    products["product_id"] = pd.to_numeric(products["product_id"], errors="coerce")
    orders["quantity"] = pd.to_numeric(orders["quantity"], errors="coerce")
    products["unit_price"] = pd.to_numeric(products["unit_price"], errors="coerce")
    orders["event_date"] = orders["event_ts"].map(_canonical_event_date)
    merged = orders.merge(products[["product_id", "unit_price"]], on="product_id", how="left")
    merged["revenue"] = merged["quantity"] * merged["unit_price"]
    expected_summary = (
        merged.groupby("event_date", as_index=False)
        .agg(order_count=("order_id", "count"), total_revenue=("revenue", "sum"))
        .sort_values("event_date")
        .reset_index(drop=True)
    )
    observed_summary = summary_df.copy().sort_values("event_date").reset_index(drop=True)
    if list(observed_summary.columns) != list(expected_summary.columns):
        return 0.0
    if len(observed_summary) != len(expected_summary):
        return 0.0
    observed_summary["total_revenue"] = pd.to_numeric(
        observed_summary["total_revenue"], errors="coerce"
    ).round(2)
    expected_summary["total_revenue"] = pd.to_numeric(
        expected_summary["total_revenue"], errors="coerce"
    ).round(2)
    matches = (observed_summary == expected_summary).all(axis=1)
    return float(matches.mean()) if len(matches) else 1.0


def task5_rollup_consistency_score(
    source_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    rollup_df: pd.DataFrame,
) -> float:
    """Score whether temporal rollup matches the repaired upstream source."""

    if not {"product_id", "quantity"}.issubset(source_df.columns):
        return 0.0
    time_column = "event_ts" if "event_ts" in source_df.columns else "observed_at"
    if time_column not in source_df.columns:
        return 0.0
    if not {"product_id", "unit_price"}.issubset(catalog_df.columns):
        return 0.0
    rollup_time_column = "hour_bucket" if "hour_bucket" in rollup_df.columns else "window_start"
    if not {rollup_time_column, "order_count", "gross_revenue"}.issubset(rollup_df.columns):
        return 0.0

    source = source_df.copy()
    catalog = catalog_df.copy()
    source["product_id"] = pd.to_numeric(source["product_id"], errors="coerce")
    catalog["product_id"] = pd.to_numeric(catalog["product_id"], errors="coerce")
    source["quantity"] = pd.to_numeric(source["quantity"], errors="coerce")
    catalog["unit_price"] = pd.to_numeric(catalog["unit_price"], errors="coerce")
    source["hour_bucket"] = source[time_column].map(_canonical_hour_bucket)
    merged = source.merge(catalog[["product_id", "unit_price"]], on="product_id", how="left")
    merged["gross_revenue"] = merged["quantity"] * merged["unit_price"]
    expected_rollup = (
        merged.groupby("hour_bucket", as_index=False)
        .agg(order_count=("order_id", "count"), gross_revenue=("gross_revenue", "sum"))
        .sort_values("hour_bucket")
        .reset_index(drop=True)
    )
    observed_rollup = rollup_df.copy().rename(columns={rollup_time_column: "hour_bucket"})
    observed_rollup = observed_rollup.sort_values("hour_bucket").reset_index(drop=True)
    if list(observed_rollup.columns) != list(expected_rollup.columns):
        return 0.0
    if len(observed_rollup) != len(expected_rollup):
        return 0.0
    observed_rollup["gross_revenue"] = pd.to_numeric(observed_rollup["gross_revenue"], errors="coerce").round(2)
    expected_rollup["gross_revenue"] = pd.to_numeric(expected_rollup["gross_revenue"], errors="coerce").round(2)
    matches = (observed_rollup == expected_rollup).all(axis=1)
    return float(matches.mean()) if len(matches) else 1.0


def _canonical_event_date(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%d/%m/%Y %H:%M",
        "%m-%d-%Y %H:%M",
        "%Y-%m-%d",
    ):
        try:
            parsed = pd.to_datetime(text, format=fmt, utc=True)
            return parsed.strftime("%Y-%m-%d")
        except Exception:
            continue
    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.notna(parsed):
        return parsed.strftime("%Y-%m-%d")
    return None


def _canonical_hour_bucket(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%d/%m/%Y %H:%M",
        "%Y-%m-%d %H:%M:%S+05:30",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            parsed = pd.to_datetime(text, format=fmt, utc=True)
            return parsed.strftime("%Y-%m-%dT%H:00:00Z")
        except Exception:
            continue
    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.notna(parsed):
        return parsed.strftime("%Y-%m-%dT%H:00:00Z")
    return None


def _accuracy(fixed_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> float:
    fixed = fixed_df.reset_index(drop=True)
    truth = ground_truth_df.reset_index(drop=True)
    if list(fixed.columns) != list(truth.columns):
        return 0.0
    if len(fixed) != len(truth):
        return 0.0
    matches = (fixed == truth).all(axis=1)
    return float(matches.mean()) if len(matches) else 1.0


def _primary_key_column(frame: pd.DataFrame) -> str | None:
    for candidate in ("transaction_id", "order_id", "customer_id", "product_id"):
        if candidate in frame.columns:
            return candidate
    return None
