from __future__ import annotations

import pandas as pd

from .shared import IncidentFixture


def load_task3_fixture(profile: str, split: str) -> IncidentFixture:
    customers_truth = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5],
            "name": ["Asha", "Ravi", "Neha", "Kabir", "Meera"],
            "email": [
                "asha@example.com",
                "ravi@example.com",
                "neha@example.com",
                "kabir@example.com",
                "meera@example.com",
            ],
            "age": [24, 31, 29, 43, 37],
        }
    )
    products_truth = pd.DataFrame(
        {
            "product_id": [10, 11, 12, 13],
            "product_name": ["Valve", "Sensor", "Pump", "Switch"],
            "unit_price": [20.0, 50.0, 80.0, 35.0],
            "category": ["hardware", "iot", "hardware", "iot"],
        }
    )
    orders_truth = pd.DataFrame(
        {
            "order_id": [201, 202, 203, 204, 205],
            "customer_id": [1, 2, 3, 4, 5],
            "product_id": [10, 11, 12, 13, 10],
            "quantity": [2, 1, 3, 2, 5],
            "order_date": [
                "2026-03-01",
                "2026-03-02",
                "2026-03-03",
                "2026-03-04",
                "2026-03-05",
            ],
        }
    )
    orders_truth = orders_truth.merge(
        products_truth[["product_id", "unit_price"]],
        on="product_id",
        how="left",
    )
    orders_truth["total_price"] = orders_truth["quantity"] * orders_truth["unit_price"]
    orders_truth = orders_truth.drop(columns=["unit_price"])

    customers_broken = customers_truth.copy()
    products_broken = products_truth.copy()
    orders_broken = orders_truth.copy()

    if profile == "customer_product_contract_drift":
        customers_broken["age"] = customers_broken["age"].astype(object)
        customers_broken.loc[2, "age"] = None
        products_broken["unit_price"] = ["$20.00", "$50.00", "80.0", "$35.00"]
        products_broken.loc[1, "category"] = " IoT "
        orders_broken["customer_id"] = orders_broken["customer_id"].astype(str)
        orders_broken.loc[4, "quantity"] = "5 units"
        orders_broken["order_date"] = ["03/01/2026", "2026-03-02", "03/03/2026", "2026-03-04", "03/05/2026"]
    elif profile == "alias_and_encoding_regression":
        customers_broken["age"] = customers_broken["age"].astype(object)
        customers_broken.loc[1, "email"] = " RAVI@EXAMPLE.COM "
        customers_broken.loc[3, "age"] = None
        products_broken = products_broken.rename(columns={"category": "product_category"})
        products_broken["unit_price"] = ["20.0", "$50.00", "$80.00", "35.0"]
        products_broken.loc[1, "product_category"] = " IoT "
        products_broken = pd.concat([products_broken, products_broken.iloc[[2]]], ignore_index=True)
        orders_broken["customer_id"] = orders_broken["customer_id"].astype(str)
        orders_broken["quantity"] = orders_broken["quantity"].astype(str)
        orders_broken.loc[0, "quantity"] = "2 units"
        orders_broken.loc[2, "order_date"] = "03/03/2026"
        orders_broken = pd.concat([orders_broken, orders_broken.iloc[[1]]], ignore_index=True)
    elif profile == "sentinel_reference_breakage":
        customers_broken["age"] = customers_broken["age"].astype(object)
        customers_broken.loc[4, "age"] = "unknown"
        products_broken["unit_price"] = ["20.0", "50.0", "999999.0", "35.0"]
        products_broken = pd.concat([products_broken, products_broken.iloc[[0]]], ignore_index=True)
        orders_broken["customer_id"] = orders_broken["customer_id"].astype(str)
        orders_broken["quantity"] = orders_broken["quantity"].astype(str)
        orders_broken.loc[1, "quantity"] = "missing"
        orders_broken.loc[3, "quantity"] = "2 units"
        orders_broken = pd.concat([orders_broken, orders_broken.iloc[[3]]], ignore_index=True)
    elif profile == "timezone_currency_consistency_incident":
        customers_broken["age"] = customers_broken["age"].astype(object)
        customers_broken.loc[0, "age"] = None
        products_broken["unit_price"] = ["$20.00", "$50.00", "80.0", "$35.00"]
        products_broken.loc[1, "category"] = " IoT "
        orders_broken["customer_id"] = orders_broken["customer_id"].astype(str)
        orders_broken["quantity"] = orders_broken["quantity"].astype(str)
        orders_broken["order_date"] = [
            "2026-03-01T00:00:00+05:30",
            "02/03/2026",
            "2026-03-03T00:00:00+05:30",
            "04/03/2026",
            "2026-03-05T00:00:00+05:30",
        ]
        orders_broken = pd.concat([orders_broken, orders_broken.iloc[[0]]], ignore_index=True)
    else:
        customers_broken["age"] = customers_broken["age"].astype(object)
        customers_broken.loc[3, "email"] = " KABIR@EXAMPLE.COM "
        customers_broken.loc[0, "age"] = None
        products_broken = products_broken.rename(columns={"category": "product_category"})
        products_broken["unit_price"] = ["$20.00", "$50.00", "999999.0", "$35.00"]
        products_broken.loc[1, "product_category"] = " IoT "
        products_broken = pd.concat([products_broken, products_broken.iloc[[1]]], ignore_index=True)
        orders_broken["customer_id"] = orders_broken["customer_id"].astype(str)
        orders_broken["quantity"] = orders_broken["quantity"].astype(str)
        orders_broken.loc[1, "quantity"] = "missing"
        orders_broken.loc[3, "quantity"] = "2 units"
        orders_broken["order_date"] = [
            "2026-03-01T00:00:00+05:30",
            "02/03/2026",
            "2026-03-03T00:00:00+05:30",
            "04/03/2026",
            "2026-03-05T00:00:00+05:30",
        ]
        orders_broken = pd.concat([orders_broken, orders_broken.iloc[[2]]], ignore_index=True)

    incident_id = f"t3-{profile}-{split}"
    metadata = {
        "incident_manifest": {
            "incident_id": incident_id,
            "dag_id": "customer_orders_daily_sync",
            "warehouse": "analytics_wh",
            "severity": "high" if split == "eval" else "medium",
            "failed_tasks": ["load_customers", "load_products", "reconcile_orders"],
            "downstream_assets": ["orders_enriched", "daily_gross_margin"],
        },
        "dag_runs": pd.DataFrame(
            [
                {"run_id": f"{incident_id}-1", "task_id": "extract_customers", "state": "success", "attempt": 1},
                {"run_id": f"{incident_id}-1", "task_id": "load_customers", "state": "failed", "attempt": 2},
                {"run_id": f"{incident_id}-1", "task_id": "load_products", "state": "failed", "attempt": 1},
                {"run_id": f"{incident_id}-1", "task_id": "reconcile_orders", "state": "upstream_failed", "attempt": 1},
            ]
        ),
        "warehouse_events": pd.DataFrame(
            [
                {"ts": "2026-03-05T09:05:00Z", "system": "warehouse", "event": "join_validation_failed"},
                {"ts": "2026-03-05T09:08:00Z", "system": "warehouse", "event": "total_price_mismatch_detected"},
                {"ts": "2026-03-05T09:10:00Z", "system": "dbt", "event": "downstream_model_blocked"},
            ]
        ),
        "trace_drift_markers": [
            "warehouse_join_validation",
            "customer_dimension_contract_drift",
            "order_total_consistency_alarm",
        ],
        "trace_dependency_health": {
            "customer_contract": "degraded",
            "product_contract": "degraded",
            "order_dependency": "cascading_breakage",
        },
        "recent_failure_counters": {
            "join_validation_failures": 1,
            "downstream_total_mismatches": int(len(orders_truth)),
            "retry_exhaustions": 1,
        },
        "queue_backlog_age_minutes": 0,
        "operational_trace_summary": "Bundled DAG-task and warehouse validation events show referential drift propagating into order totals.",
    }
    return IncidentFixture(
        broken_tables={
            "orders": orders_broken,
            "customers": customers_broken,
            "products": products_broken,
        },
        ground_truth_tables={
            "orders": orders_truth,
            "customers": customers_truth,
            "products": products_truth,
        },
        metadata=metadata,
    )
