from __future__ import annotations

import pandas as pd

from .shared import score_single_table


def calculation_mismatch_count(
    orders_df: pd.DataFrame,
    products_df: pd.DataFrame,
) -> int:
    required_order_columns = {"product_id", "quantity", "total_price"}
    required_product_columns = {"product_id", "unit_price"}
    if not required_order_columns.issubset(orders_df.columns):
        return 0
    if not required_product_columns.issubset(products_df.columns):
        return 0

    orders = orders_df.copy()
    products = products_df[["product_id", "unit_price"]].copy()
    orders["product_id"] = pd.to_numeric(orders["product_id"], errors="coerce")
    orders["quantity"] = pd.to_numeric(orders["quantity"], errors="coerce")
    orders["total_price"] = pd.to_numeric(orders["total_price"], errors="coerce")
    products["product_id"] = pd.to_numeric(products["product_id"], errors="coerce")
    products["unit_price"] = pd.to_numeric(products["unit_price"], errors="coerce")
    merged = orders.merge(products, on="product_id", how="left")
    expected = (merged["quantity"] * merged["unit_price"]).round(2)
    return int((expected.round(2) != merged["total_price"].round(2)).sum())


def task3_dependency_score(
    orders_df: pd.DataFrame,
    products_df: pd.DataFrame,
) -> float:
    mismatch_count = calculation_mismatch_count(orders_df, products_df)
    if len(orders_df) == 0:
        return 0.0
    return max(0.0, 1.0 - (mismatch_count / len(orders_df)))


def score_task3(
    fixed_tables: dict[str, pd.DataFrame],
    ground_truth_tables: dict[str, pd.DataFrame],
    expected_types: dict[str, dict[str, str]],
) -> tuple[float, dict[str, dict[str, float]]]:
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
