from __future__ import annotations

import numpy as np
import pandas as pd

from .shared import (
    FORMAL_TASK_SPECS,
    SYNTHETIC_DATA_NOTES,
    TASK_OBJECTIVE_WEIGHTS,
    Scenario,
    expected_types,
    patterns_for_profile,
    sample_profile,
)


def generate_task5(
    rng: np.random.Generator,
    seed: int | None,
    split: str,
) -> Scenario:
    profile = sample_profile(5, split, rng)
    catalog_truth = pd.DataFrame(
        {
            "product_id": [601, 602, 603, 604],
            "product_name": ["Valve", "Sensor", "Pump", "Controller"],
            "unit_price": [18.0, 52.0, 77.0, 112.0],
            "category": ["hardware", "iot", "hardware", "iot"],
        }
    )
    source_truth = pd.DataFrame(
        {
            "order_id": [11001, 11002, 11003, 11004, 11005, 11006, 11007, 11008, 11009, 11010],
            "batch_id": ["t1", "t1", "t1", "t2", "t2", "t2", "t3", "t3", "t4", "t4"],
            "product_id": [601, 602, 604, 603, 601, 602, 604, 603, 601, 604],
            "quantity": [2, 1, 3, 2, 4, 1, 2, 3, 1, 5],
            "event_ts": [
                "2026-03-29T10:00:00Z",
                "2026-03-29T10:30:00Z",
                "2026-03-29T11:00:00Z",
                "2026-03-29T12:00:00Z",
                "2026-03-29T12:20:00Z",
                "2026-03-29T13:00:00Z",
                "2026-03-29T14:00:00Z",
                "2026-03-29T14:20:00Z",
                "2026-03-29T15:00:00Z",
                "2026-03-29T15:30:00Z",
            ],
        }
    )
    source_truth = source_truth.merge(
        catalog_truth[["product_id", "unit_price"]],
        on="product_id",
        how="left",
    )
    source_truth["gross_revenue"] = source_truth["quantity"] * source_truth["unit_price"]
    source_truth = source_truth.drop(columns=["unit_price"])
    rollup_truth = (
        source_truth.assign(
            hour_bucket=pd.to_datetime(source_truth["event_ts"]).dt.strftime("%Y-%m-%dT%H:00:00Z")
        )
        .groupby("hour_bucket", as_index=False)
        .agg(order_count=("order_id", "count"), gross_revenue=("gross_revenue", "sum"))
    )

    pending_orders = source_truth[source_truth["batch_id"].isin(["t3", "t4"])].copy()
    visible_orders = source_truth[source_truth["batch_id"].isin(["t1", "t2"])].copy()
    source_broken = visible_orders.copy()
    catalog_broken = catalog_truth.copy()
    rollup_broken = rollup_truth[rollup_truth["hour_bucket"] < "2026-03-29T14:00:00Z"].copy()

    source_broken["product_id"] = source_broken["product_id"].astype(str)
    source_broken["quantity"] = source_broken["quantity"].astype(object)
    source_broken["event_ts"] = source_broken["event_ts"].astype(object)
    source_broken["gross_revenue"] = (visible_orders["gross_revenue"] * 0.83).round(2).astype(object)

    if profile in {"schema_evolution_backfill_recovery", "temporal_open_world_shift_incident", "heldout_temporal_incident_family"}:
        source_broken = source_broken.rename(columns={"event_ts": "observed_at"})
    time_column = "observed_at" if "observed_at" in source_broken.columns else "event_ts"
    drift_rows = rng.choice(len(source_broken), size=min(3, len(source_broken)), replace=False)
    for index, row in enumerate(drift_rows):
        ts = pd.to_datetime(visible_orders.iloc[row]["event_ts"])
        if index % 2 == 0:
            source_broken.iloc[row, source_broken.columns.get_loc(time_column)] = ts.strftime("%d/%m/%Y %H:%M")
        else:
            source_broken.iloc[row, source_broken.columns.get_loc(time_column)] = ts.strftime("%Y-%m-%d %H:%M:%S+05:30")
    quantity_rows = rng.choice(len(source_broken), size=min(3, len(source_broken)), replace=False)
    for row in quantity_rows:
        quantity = visible_orders.iloc[row]["quantity"]
        source_broken.iloc[row, source_broken.columns.get_loc("quantity")] = (
            "missing" if row == quantity_rows[0] else f"{quantity} units"
        )

    catalog_broken["unit_price"] = catalog_broken["unit_price"].astype(object)
    formatted_rows = rng.choice(len(catalog_broken), size=min(3, len(catalog_broken)), replace=False)
    for row in formatted_rows:
        value = catalog_truth.iloc[row]["unit_price"]
        catalog_broken.iloc[row, catalog_broken.columns.get_loc("unit_price")] = (
            f"${value:,.2f}" if row % 2 == 0 else f"{int(value * 100)} cents"
        )
    if profile in {"schema_evolution_backfill_recovery", "heldout_temporal_incident_family"}:
        catalog_broken = catalog_broken.rename(columns={"category": "product_segment"})
    category_column = "product_segment" if "product_segment" in catalog_broken.columns else "category"
    catalog_broken.iloc[1, catalog_broken.columns.get_loc(category_column)] = " IoT "
    catalog_broken = pd.concat([catalog_broken, catalog_broken.iloc[[2]]], ignore_index=True)

    if profile in {"temporal_rollup_backfill_incident", "temporal_open_world_shift_incident", "heldout_temporal_incident_family"}:
        rollup_broken = rollup_broken.rename(columns={"hour_bucket": "window_start"})
    rollup_time_column = "window_start" if "window_start" in rollup_broken.columns else "hour_bucket"
    if profile == "heldout_temporal_incident_family":
        rollup_broken = rollup_broken.rename(columns={"gross_revenue": "gross_sales"})
    if len(rollup_broken) > 0:
        rollup_broken.iloc[0, rollup_broken.columns.get_loc(rollup_time_column)] = "29/03/2026 10:00"
    rollup_revenue_column = "gross_sales" if "gross_sales" in rollup_broken.columns else "gross_revenue"
    rollup_broken[rollup_revenue_column] = (rollup_broken[rollup_revenue_column] * 0.88).round(2)

    backlog_rows = len(pending_orders)
    required_resource_level = 3 if backlog_rows >= 4 else 2
    freshness_lag_minutes = 180 if split == "eval" else 120
    backlog_age_minutes = 240 if split == "train" else 360

    return Scenario(
        task_id=5,
        seed=seed,
        broken_tables={
            "source_orders": source_broken,
            "catalog": catalog_broken,
            "hourly_rollup": rollup_broken,
        },
        ground_truth_tables={
            "source_orders": source_truth,
            "catalog": catalog_truth,
            "hourly_rollup": rollup_truth,
        },
        expected_types={
            "source_orders": expected_types(source_truth),
            "catalog": expected_types(catalog_truth),
            "hourly_rollup": expected_types(rollup_truth),
        },
        active_table="source_orders",
        split=split,
        metadata={
            "pending_orders": pending_orders,
            "backlog_rows": backlog_rows,
            "queue_backlog_age_minutes": backlog_age_minutes,
            "freshness_lag_minutes": freshness_lag_minutes,
            "resource_level": 1,
            "required_resource_level": required_resource_level,
            "pending_batches": 2 if backlog_rows > 0 else 0,
            "downstream_stale": True,
            "workload_pressure": 0.95 if split == "eval" else 0.8,
            "scenario_profile": profile,
            "open_world_patterns": patterns_for_profile(profile),
            "synthetic_data_notes": SYNTHETIC_DATA_NOTES,
            "heldout_profile_family": profile == "heldout_temporal_incident_family",
            "adaptation_target": "Recover an unseen temporal profile family in one episode.",
            "task_spec": FORMAL_TASK_SPECS[5],
            "tradeoff_weights": TASK_OBJECTIVE_WEIGHTS[5],
            "recent_failure_counters": {
                "late_correction_failures": 2,
                "rollup_refresh_failures": 1,
                "schema_migration_regressions": 1,
            },
        },
    )
