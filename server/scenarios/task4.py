from __future__ import annotations

import numpy as np
import pandas as pd

from .shared import SYNTHETIC_DATA_NOTES, Scenario, expected_types, patterns_for_profile, sample_profile


def generate_task4(
    rng: np.random.Generator,
    seed: int | None,
    split: str,
) -> Scenario:
    profile = sample_profile(4, split, rng)
    products_truth = pd.DataFrame(
        {
            "product_id": [401, 402, 403, 404],
            "product_name": ["Valve", "Sensor", "Pump", "Controller"],
            "unit_price": [15.0, 48.0, 73.0, 105.0],
            "category": ["hardware", "iot", "hardware", "iot"],
        }
    )
    orders_truth = pd.DataFrame(
        {
            "order_id": [9001, 9002, 9003, 9004, 9005, 9006, 9007, 9008],
            "batch_id": ["b1", "b1", "b1", "b2", "b2", "b3", "b3", "b3"],
            "product_id": [401, 402, 404, 403, 401, 404, 402, 403],
            "quantity": [2, 1, 3, 2, 5, 1, 2, 4],
            "event_ts": [
                "2026-03-28T10:00:00Z",
                "2026-03-28T10:05:00Z",
                "2026-03-28T10:15:00Z",
                "2026-03-29T09:10:00Z",
                "2026-03-29T10:30:00Z",
                "2026-03-30T08:45:00Z",
                "2026-03-30T09:00:00Z",
                "2026-03-30T09:30:00Z",
            ],
        }
    )
    orders_truth = orders_truth.merge(
        products_truth[["product_id", "unit_price"]],
        on="product_id",
        how="left",
    )
    orders_truth["revenue"] = orders_truth["quantity"] * orders_truth["unit_price"]
    orders_truth = orders_truth.drop(columns=["unit_price"])

    summary_truth = (
        orders_truth.assign(event_date=pd.to_datetime(orders_truth["event_ts"]).dt.strftime("%Y-%m-%d"))
        .groupby("event_date", as_index=False)
        .agg(order_count=("order_id", "count"), total_revenue=("revenue", "sum"))
    )

    pending_orders = orders_truth[orders_truth["batch_id"] == "b3"].copy()
    visible_orders = orders_truth[orders_truth["batch_id"] != "b3"].copy()
    orders_broken = visible_orders.copy()
    products_broken = products_truth.copy()
    summary_broken = summary_truth[summary_truth["event_date"] != "2026-03-30"].copy()

    orders_broken["product_id"] = orders_broken["product_id"].astype(str)
    orders_broken["quantity"] = orders_broken["quantity"].astype(str)
    orders_broken["event_ts"] = orders_broken["event_ts"].astype(object)
    drift_rows = rng.choice(len(orders_broken), size=min(3, len(orders_broken)), replace=False)
    for index, row in enumerate(drift_rows):
        ts = pd.to_datetime(visible_orders.iloc[row]["event_ts"])
        if profile in {"timezone_alias_burst_incident", "mixed_operational_recovery_incident"} and index == 0:
            orders_broken.iloc[row, orders_broken.columns.get_loc("event_ts")] = ts.strftime("%Y-%m-%d %H:%M:%S+05:30")
        elif index % 2 == 0:
            orders_broken.iloc[row, orders_broken.columns.get_loc("event_ts")] = ts.strftime("%d/%m/%Y %H:%M")
        else:
            orders_broken.iloc[row, orders_broken.columns.get_loc("event_ts")] = ts.strftime("%m-%d-%Y %H:%M")
    if profile in {"schema_alias_unit_regression", "timezone_alias_burst_incident", "mixed_operational_recovery_incident"}:
        orders_broken = orders_broken.rename(columns={"event_ts": "event_time"})

    if split == "eval" or profile in {"mixed_operational_recovery_incident"} or rng.random() < 0.7:
        quantity_rows = rng.choice(len(orders_broken), size=min(2, len(orders_broken)), replace=False)
        for row in quantity_rows:
            value = visible_orders.iloc[row]["quantity"]
            orders_broken.iloc[row, orders_broken.columns.get_loc("quantity")] = f"{value} units"

    products_broken["unit_price"] = products_broken["unit_price"].astype(object)
    formatted_rows = rng.choice(len(products_broken), size=min(3, len(products_broken)), replace=False)
    for row in formatted_rows:
        value = products_truth.iloc[row]["unit_price"]
        if row % 2 == 0:
            products_broken.iloc[row, products_broken.columns.get_loc("unit_price")] = f"${value:,.2f}"
        else:
            products_broken.iloc[row, products_broken.columns.get_loc("unit_price")] = f"{int(value * 100)} cents"
    if split == "eval" or profile in {"schema_alias_unit_regression", "mixed_operational_recovery_incident"} or rng.random() < 0.6:
        products_broken.iloc[1, products_broken.columns.get_loc("category")] = " IoT "
    if profile in {"schema_alias_unit_regression", "mixed_operational_recovery_incident"}:
        products_broken = products_broken.rename(columns={"category": "product_segment"})

    summary_broken["event_date"] = summary_broken["event_date"].astype(object)
    if len(summary_broken) > 0:
        first_date = pd.to_datetime(summary_broken.iloc[0]["event_date"])
        summary_broken.iloc[0, summary_broken.columns.get_loc("event_date")] = first_date.strftime("%d/%m/%Y")
    summary_broken["total_revenue"] = (summary_broken["total_revenue"] * 0.92).round(2)
    if profile in {"stale_summary_oncall_recovery", "mixed_operational_recovery_incident"}:
        summary_broken = summary_broken.rename(columns={"event_date": "business_date"})

    backlog_rows = len(pending_orders)
    required_resource_level = 2 if backlog_rows <= 2 else 3
    freshness_lag_minutes = 90 if split == "train" else 150
    if profile in {"timezone_alias_burst_incident", "mixed_operational_recovery_incident"}:
        freshness_lag_minutes += 30
    backlog_age_minutes = 165 if split == "train" else 240

    return Scenario(
        task_id=4,
        seed=seed,
        broken_tables={
            "orders": orders_broken,
            "products": products_broken,
            "daily_summary": summary_broken,
        },
        ground_truth_tables={
            "orders": orders_truth,
            "products": products_truth,
            "daily_summary": summary_truth,
        },
        expected_types={
            "orders": expected_types(orders_truth),
            "products": expected_types(products_truth),
            "daily_summary": expected_types(summary_truth),
        },
        active_table="orders",
        split=split,
        metadata={
            "pending_orders": pending_orders,
            "backlog_rows": backlog_rows,
            "queue_backlog_age_minutes": backlog_age_minutes,
            "freshness_lag_minutes": freshness_lag_minutes,
            "resource_level": 1,
            "required_resource_level": required_resource_level,
            "pending_batches": 1 if backlog_rows > 0 else 0,
            "downstream_stale": True,
            "workload_pressure": 0.9 if split == "eval" else 0.75,
            "scenario_profile": profile,
            "open_world_patterns": patterns_for_profile(profile),
            "synthetic_data_notes": SYNTHETIC_DATA_NOTES,
            "recent_failure_counters": {
                "incremental_load_failures": 1,
                "summary_refresh_failures": 1,
                "resource_scale_attempts": 1 if required_resource_level > 1 else 0,
            },
        },
    )
