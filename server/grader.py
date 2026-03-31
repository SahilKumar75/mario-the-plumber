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
