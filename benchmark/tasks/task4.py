from __future__ import annotations

import pandas as pd

from .shared import _accuracy, _canonical_event_date, score_single_table


def task4_batch_completeness_score(
    orders_df: pd.DataFrame,
    ground_truth_orders_df: pd.DataFrame,
    backlog_rows: int,
) -> float:
    if backlog_rows <= 0:
        return 1.0
    recovered_ratio = min(len(orders_df) / max(len(ground_truth_orders_df), 1), 1.0)
    backlog_penalty = min(backlog_rows / max(len(ground_truth_orders_df), 1), 1.0)
    return max(0.0, recovered_ratio - (0.5 * backlog_penalty))


def task4_summary_consistency_score(
    orders_df: pd.DataFrame,
    products_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> float:
    required_order_columns = {"order_id", "product_id", "quantity", "event_ts"}
    required_product_columns = {"product_id", "unit_price"}
    required_summary_columns = {"event_date", "order_count", "total_revenue"}
    if not required_order_columns.issubset(orders_df.columns):
        return 0.0
    if not required_product_columns.issubset(products_df.columns):
        return 0.0
    if not required_summary_columns.issubset(summary_df.columns):
        return 0.0

    orders = orders_df.copy()
    products = products_df[["product_id", "unit_price"]].copy()
    orders["product_id"] = pd.to_numeric(orders["product_id"], errors="coerce")
    orders["quantity"] = pd.to_numeric(orders["quantity"], errors="coerce")
    products["product_id"] = pd.to_numeric(products["product_id"], errors="coerce")
    products["unit_price"] = pd.to_numeric(products["unit_price"], errors="coerce")
    orders["event_date"] = orders["event_ts"].map(_canonical_event_date)
    merged = orders.merge(products, on="product_id", how="left")
    merged["total_revenue"] = merged["quantity"] * merged["unit_price"]
    expected = (
        merged.groupby("event_date", as_index=False)
        .agg(order_count=("order_id", "count"), total_revenue=("total_revenue", "sum"))
        .sort_values("event_date")
        .reset_index(drop=True)
    )
    actual = summary_df.copy().sort_values("event_date").reset_index(drop=True)
    return _accuracy(expected, actual)


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

    data_quality = (
        (0.40 * orders_score)
        + (0.20 * products_score)
        + (0.20 * summary_score)
    ) / 0.80
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

    score = round(
        (0.45 * data_quality)
        + (0.20 * freshness_score)
        + (0.15 * batch_score)
        + (0.10 * resource_efficiency)
        + (0.10 * summary_consistency),
        4,
    )
    return score, {
        "orders": orders_breakdown,
        "products": products_breakdown,
        "daily_summary": summary_breakdown,
        "pipeline": {
            "data_quality": round(float(data_quality), 4),
            "backlog": round(batch_score, 4),
            "freshness": round(freshness_score, 4),
            "summary_consistency": round(summary_consistency, 4),
            "resource_efficiency": round(resource_efficiency, 4),
        },
    }
