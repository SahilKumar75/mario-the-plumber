from __future__ import annotations

import pandas as pd

from .shared import IncidentFixture


def load_task4_fixture(profile: str, split: str) -> IncidentFixture:
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
    orders_truth = orders_truth.merge(products_truth[["product_id", "unit_price"]], on="product_id", how="left")
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

    if profile == "late_batch_resource_incident":
        orders_broken["product_id"] = orders_broken["product_id"].astype(str)
        orders_broken["quantity"] = ["2 units", "1", "3", "2", "5 units"]
        orders_broken["event_ts"] = [
            "28/03/2026 10:00",
            "2026-03-28T10:05:00Z",
            "03-28-2026 10:15",
            "2026-03-29T09:10:00Z",
            "03-29-2026 10:30",
        ]
        products_broken["unit_price"] = ["$15.00", "48.0", "$73.00", "10500 cents"]
        summary_broken["total_revenue"] = (summary_broken["total_revenue"] * 0.92).round(2)
    elif profile == "schema_alias_unit_regression":
        orders_broken["product_id"] = orders_broken["product_id"].astype(str)
        orders_broken["quantity"] = ["2", "1 units", "3", "2", "5"]
        orders_broken = orders_broken.rename(columns={"event_ts": "event_time"})
        orders_broken["event_time"] = [
            "28/03/2026 10:00",
            "2026-03-28T10:05:00Z",
            "03-28-2026 10:15",
            "2026-03-29T09:10:00Z",
            "2026-03-29T10:30:00Z",
        ]
        products_broken["unit_price"] = ["$15.00", "4800 cents", "$73.00", "10500 cents"]
        products_broken = products_broken.rename(columns={"category": "product_segment"})
        products_broken.loc[1, "product_segment"] = " IoT "
        summary_broken["total_revenue"] = (summary_broken["total_revenue"] * 0.92).round(2)
    elif profile == "stale_summary_oncall_recovery":
        orders_broken["product_id"] = orders_broken["product_id"].astype(str)
        orders_broken["quantity"] = ["2", "1", "3 units", "2", "5 units"]
        orders_broken["event_ts"] = [
            "2026-03-28T10:00:00Z",
            "2026-03-28T10:05:00Z",
            "03-28-2026 10:15",
            "2026-03-29T09:10:00Z",
            "03-29-2026 10:30",
        ]
        products_broken["unit_price"] = ["$15.00", "$48.00", "73.0", "$105.00"]
        summary_broken = summary_broken.rename(columns={"event_date": "business_date"})
        summary_broken["total_revenue"] = (summary_broken["total_revenue"] * 0.92).round(2)
    elif profile == "timezone_alias_burst_incident":
        orders_broken["product_id"] = orders_broken["product_id"].astype(str)
        orders_broken["quantity"] = ["2 units", "1", "3", "2", "5 units"]
        orders_broken = orders_broken.rename(columns={"event_ts": "event_time"})
        orders_broken["event_time"] = [
            "2026-03-28 15:30:00+05:30",
            "2026-03-28T10:05:00Z",
            "28/03/2026 10:15",
            "2026-03-29T09:10:00Z",
            "03-29-2026 10:30",
        ]
        products_broken["unit_price"] = ["$15.00", "4800 cents", "$73.00", "$105.00"]
        summary_broken["total_revenue"] = (summary_broken["total_revenue"] * 0.92).round(2)
    else:
        orders_broken["product_id"] = orders_broken["product_id"].astype(str)
        orders_broken["quantity"] = ["2 units", "1", "3", "2", "5 units"]
        orders_broken = orders_broken.rename(columns={"event_ts": "event_time"})
        orders_broken["event_time"] = [
            "2026-03-28 15:30:00+05:30",
            "2026-03-28T10:05:00Z",
            "28/03/2026 10:15",
            "2026-03-29T09:10:00Z",
            "03-29-2026 10:30",
        ]
        products_broken["unit_price"] = ["$15.00", "4800 cents", "$73.00", "10500 cents"]
        products_broken = products_broken.rename(columns={"category": "product_segment"})
        products_broken.loc[1, "product_segment"] = " IoT "
        summary_broken = summary_broken.rename(columns={"event_date": "business_date"})
        summary_broken["total_revenue"] = (summary_broken["total_revenue"] * 0.92).round(2)

    backlog_rows = len(pending_orders)
    required_resource_level = 2 if backlog_rows <= 2 else 3
    freshness_lag_minutes = 90 if split == "train" else 150
    if profile in {"timezone_alias_burst_incident", "mixed_operational_recovery_incident"}:
        freshness_lag_minutes += 30
    backlog_age_minutes = 165 if split == "train" else 240
    incident_id = f"t4-{profile}-{split}"
    metadata = {
        "pending_orders": pending_orders,
        "backlog_rows": backlog_rows,
        "queue_backlog_age_minutes": backlog_age_minutes,
        "freshness_lag_minutes": freshness_lag_minutes,
        "resource_level": 1,
        "required_resource_level": required_resource_level,
        "pending_batches": 1 if backlog_rows > 0 else 0,
        "downstream_stale": True,
        "workload_pressure": 0.9 if split == "eval" else 0.75,
        "incident_manifest": {
            "incident_id": incident_id,
            "dag_id": "incremental_orders_refresh",
            "warehouse": "ops_wh",
            "severity": "critical" if split == "eval" else "high",
            "failed_tasks": ["load_incremental_orders", "refresh_daily_summary"],
            "downstream_assets": ["daily_summary", "sales_dashboard"],
        },
        "dag_runs": pd.DataFrame(
            [
                {"run_id": f"{incident_id}-1", "task_id": "extract_orders", "state": "success", "attempt": 1},
                {"run_id": f"{incident_id}-1", "task_id": "load_incremental_orders", "state": "failed", "attempt": 2},
                {"run_id": f"{incident_id}-1", "task_id": "refresh_daily_summary", "state": "failed", "attempt": 1},
            ]
        ),
        "warehouse_events": pd.DataFrame(
            [
                {"ts": "2026-03-30T09:35:00Z", "system": "airflow", "event": "incremental_batch_retry_exhausted"},
                {"ts": "2026-03-30T09:37:00Z", "system": "warehouse", "event": "summary_table_stale"},
                {"ts": "2026-03-30T09:40:00Z", "system": "autoscaler", "event": "worker_pool_under_provisioned"},
            ]
        ),
        "trace_drift_markers": [
            "incremental_replay_backlog",
            "summary_refresh_staleness",
            "worker_pool_pressure",
        ],
        "trace_dependency_health": {
            "incremental_backlog": "pending_replay",
            "summary_state": "stale",
            "recovery_gate": "recovery_incomplete",
        },
        "recent_failure_counters": {
            "incremental_load_failures": 1,
            "summary_refresh_failures": 1,
            "resource_scale_attempts": 1 if required_resource_level > 1 else 0,
        },
        "operational_trace_summary": "Bundled airflow-style task runs and warehouse freshness alarms show an on-call incremental replay incident.",
    }
    return IncidentFixture(
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
        metadata=metadata,
    )
