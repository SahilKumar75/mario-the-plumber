"""Deterministic scoring and reward helpers for PipelineDoctor."""

from __future__ import annotations

from datetime import datetime

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
            "data_quality": round(data_quality, 4),
            "batch_completeness": round(batch_score, 4),
            "backlog": round(batch_score, 4),
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
        reward -= 0.05
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
    """Structured explanation of the scalar reward."""

    progress = round(0.5 * (score_after - score_before), 4)
    step_cost = -0.001
    invalid_penalty = -0.05 if not action_valid else 0.0
    terminal_bonus = 1.0 if done and success else 0.0
    terminal_penalty = -0.5 if done and not success else 0.0
    total = round(progress + step_cost + invalid_penalty + terminal_bonus + terminal_penalty, 4)
    return {
        "progress": progress,
        "step_cost": step_cost,
        "invalid_penalty": invalid_penalty,
        "terminal_bonus": terminal_bonus,
        "terminal_penalty": terminal_penalty,
        "total": total,
    }


def calculation_mismatch_count(
    orders_df: pd.DataFrame,
    products_df: pd.DataFrame,
) -> int:
    """Count orders whose total_price no longer matches quantity * unit_price."""

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
    """Dependency consistency between orders.total_price and products.unit_price."""

    mismatch_count = calculation_mismatch_count(orders_df, products_df)
    if len(orders_df) == 0:
        return 0.0
    return max(0.0, 1.0 - (mismatch_count / len(orders_df)))


def task4_batch_completeness_score(
    orders_df: pd.DataFrame,
    ground_truth_orders_df: pd.DataFrame,
    backlog_rows: int,
) -> float:
    """Measure how much of the delayed incremental batch has been recovered."""

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
    """Check whether the downstream daily summary matches current upstream tables."""

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


def task5_rollup_consistency_score(
    source_orders_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    rollup_df: pd.DataFrame,
) -> float:
    """Check whether the hourly rollup matches current source_orders and catalog tables."""

    required_source_columns = {"order_id", "product_id", "quantity", "event_ts"}
    required_catalog_columns = {"product_id", "unit_price"}
    required_rollup_columns = {"hour_bucket", "order_count", "gross_revenue"}
    if not required_source_columns.issubset(source_orders_df.columns):
        return 0.0
    if not required_catalog_columns.issubset(catalog_df.columns):
        return 0.0
    if not required_rollup_columns.issubset(rollup_df.columns):
        return 0.0

    source = source_orders_df.copy()
    catalog = catalog_df[["product_id", "unit_price"]].copy()
    source["product_id"] = pd.to_numeric(source["product_id"], errors="coerce")
    source["quantity"] = pd.to_numeric(source["quantity"], errors="coerce")
    source["event_ts"] = source["event_ts"].map(_coerce_utc_timestamp)
    catalog["product_id"] = pd.to_numeric(catalog["product_id"], errors="coerce")
    catalog["unit_price"] = pd.to_numeric(catalog["unit_price"], errors="coerce")
    merged = source.merge(catalog, on="product_id", how="left")
    merged["gross_revenue"] = merged["quantity"] * merged["unit_price"]
    merged["hour_bucket"] = merged["event_ts"].dt.strftime("%Y-%m-%dT%H:00:00Z")
    expected = (
        merged.groupby("hour_bucket", as_index=False)
        .agg(order_count=("order_id", "count"), gross_revenue=("gross_revenue", "sum"))
        .sort_values("hour_bucket")
        .reset_index(drop=True)
    )
    actual = rollup_df.copy().sort_values("hour_bucket").reset_index(drop=True)
    if "hour_bucket" in actual.columns:
        actual["hour_bucket"] = actual["hour_bucket"].map(_canonical_hour_bucket)
    if "hour_bucket" in expected.columns:
        expected["hour_bucket"] = expected["hour_bucket"].map(_canonical_hour_bucket)
    return _accuracy(expected, actual)


def _canonical_event_date(value: object) -> str | None:
    if pd.isna(value):
        return None
    parsed = _coerce_utc_timestamp(value)
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d")


def _canonical_hour_bucket(value: object) -> str | None:
    if pd.isna(value):
        return None
    parsed = _coerce_utc_timestamp(value)
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%dT%H:00:00Z")


def _accuracy(fixed_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> float:
    if list(fixed_df.columns) != list(ground_truth_df.columns):
        return 0.0
    if len(fixed_df) != len(ground_truth_df):
        return 0.0
    if len(fixed_df) == 0:
        return 1.0
    fixed = fixed_df.astype(object).where(fixed_df.notna(), "__nan__")
    ground_truth = ground_truth_df.astype(object).where(ground_truth_df.notna(), "__nan__")
    matches = (fixed == ground_truth).all(axis=1)
    return float(matches.mean())


def _coerce_utc_timestamp(value: object) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT

    text = str(value).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%d/%m/%Y %H:%M", "%m-%d-%Y %H:%M", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(text, fmt)
            return pd.Timestamp(parsed, tz="UTC")
        except (TypeError, ValueError):
            continue

    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.isna(parsed):
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.isna(parsed):
            return pd.NaT
        if parsed.tzinfo is None:
            return parsed.tz_localize("UTC")
        return parsed.tz_convert("UTC")
    return parsed


def _primary_key_column(frame: pd.DataFrame) -> str | None:
    for candidate in ("transaction_id", "order_id", "customer_id", "product_id"):
        if candidate in frame.columns:
            return candidate
    return None
