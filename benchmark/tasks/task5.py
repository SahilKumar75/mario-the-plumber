from __future__ import annotations

import pandas as pd

from .shared import _accuracy, _canonical_hour_bucket, _coerce_utc_timestamp, score_single_table
from .task4 import task4_batch_completeness_score


def task5_rollup_consistency_score(
    source_orders_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    rollup_df: pd.DataFrame,
) -> float:
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


def task5_temporal_closure_score(
    source_orders_df: pd.DataFrame,
    rollup_df: pd.DataFrame,
    incident_manifest: dict[str, object] | None,
) -> float:
    if not incident_manifest:
        return 1.0
    if "order_id" not in source_orders_df.columns:
        return 0.0
    time_column = next(
        (column for column in ("event_ts", "observed_at", "source_event_ts", "source_time_utc") if column in source_orders_df.columns),
        None,
    )
    bucket_column = next((column for column in ("hour_bucket", "window_start") if column in rollup_df.columns), None)
    if time_column is None or bucket_column is None:
        return 0.0

    required_order_ids = {
        int(value)
        for value in incident_manifest.get("late_correction_order_ids", [])
        if str(value).strip()
    }
    required_buckets = {
        _canonical_hour_bucket(str(value))
        for value in incident_manifest.get("affected_hour_buckets", [])
        if str(value).strip()
    }
    expected_watermark = _coerce_utc_timestamp(incident_manifest.get("expected_watermark_after_replay"))

    source = source_orders_df.copy()
    source["order_id"] = pd.to_numeric(source["order_id"], errors="coerce")
    source["event_ts"] = source[time_column].map(_coerce_utc_timestamp)
    actual_order_ids = {int(value) for value in source["order_id"].dropna().tolist()}
    actual_watermark = source["event_ts"].dropna().max()
    actual_buckets = {
        _canonical_hour_bucket(str(value))
        for value in rollup_df[bucket_column].dropna().tolist()
        if str(value).strip()
    }

    order_score = 1.0 if not required_order_ids else len(actual_order_ids & required_order_ids) / len(required_order_ids)
    watermark_score = 1.0 if expected_watermark is None else float(pd.notna(actual_watermark) and actual_watermark >= expected_watermark)
    bucket_score = 1.0 if not required_buckets else len(actual_buckets & required_buckets) / len(required_buckets)
    return round((order_score + watermark_score + bucket_score) / 3.0, 4)


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
    incident_manifest: dict[str, object] | None = None,
) -> tuple[float, dict[str, dict[str, float]]]:
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
    temporal_closure = task5_temporal_closure_score(
        fixed_tables["source_orders"],
        fixed_tables["hourly_rollup"],
        incident_manifest,
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
        + (0.15 * temporal_backfill)
        + (0.15 * rollup_consistency)
        + (0.10 * temporal_closure)
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
            "temporal_closure": round(temporal_closure, 4),
            "freshness": round(freshness, 4),
            "resource_efficiency": round(resource_efficiency, 4),
            "data_quality": round(data_quality, 4),
        },
    }
