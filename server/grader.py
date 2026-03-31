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
