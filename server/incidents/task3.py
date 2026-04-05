from __future__ import annotations

import pandas as pd

from .shared import IncidentFixture


def load_task3_fixture(profile: str, split: str) -> IncidentFixture:
    customers_truth = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "name": ["Asha", "Ravi", "Neha", "Kabir", "Meera", "Ishaan", "Priya", "Arjun", "Farah", "Vikram", "Nina", "Dev"],
            "email": [
                "asha@example.com",
                "ravi@example.com",
                "neha@example.com",
                "kabir@example.com",
                "meera@example.com",
                "ishaan@example.com",
                "priya@example.com",
                "arjun@example.com",
                "farah@example.com",
                "vikram@example.com",
                "nina@example.com",
                "dev@example.com",
            ],
            "age": [24, 31, 29, 43, 37, 34, 28, 41, 32, 39, 27, 45],
        }
    )
    products_truth = pd.DataFrame(
        {
            "product_id": [10, 11, 12, 13, 14, 15, 16],
            "product_name": ["Valve", "Sensor", "Pump", "Switch", "Gateway", "Relay", "Transmitter"],
            "unit_price": [20.0, 50.0, 80.0, 35.0, 64.0, 44.0, 91.0],
            "category": ["hardware", "iot", "hardware", "iot", "iot", "ops", "industrial"],
        }
    )
    orders_truth = pd.DataFrame(
        {
            "order_id": [201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216],
            "customer_id": [1, 2, 3, 4, 5, 6, 7, 8, 2, 5, 9, 10, 11, 12, 4, 7],
            "product_id": [10, 11, 12, 13, 10, 14, 12, 11, 14, 13, 15, 16, 10, 15, 16, 11],
            "quantity": [2, 1, 3, 2, 5, 2, 4, 1, 3, 2, 2, 1, 6, 3, 2, 4],
            "order_date": [
                "2026-03-01",
                "2026-03-02",
                "2026-03-03",
                "2026-03-04",
                "2026-03-05",
                "2026-03-06",
                "2026-03-07",
                "2026-03-08",
                "2026-03-09",
                "2026-03-10",
                "2026-03-11",
                "2026-03-12",
                "2026-03-13",
                "2026-03-14",
                "2026-03-15",
                "2026-03-16",
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

    profile_manifest = {
        "customer_product_contract_drift": {
            "profile_family": "familiar_referential",
            "novelty_axes": ["currency_drift", "date_drift", "dependency_breakage"],
        },
        "alias_and_encoding_regression": {
            "profile_family": "familiar_referential",
            "novelty_axes": ["schema_alias", "encoding_drift", "dependency_breakage"],
        },
        "sentinel_reference_breakage": {
            "profile_family": "familiar_referential",
            "novelty_axes": ["sentinel_values", "missing_values", "dependency_breakage"],
        },
        "timezone_currency_consistency_incident": {
            "profile_family": "familiar_referential",
            "novelty_axes": ["timezone_drift", "currency_drift", "dependency_breakage"],
        },
        "cascading_reference_outage": {
            "profile_family": "familiar_referential",
            "novelty_axes": ["schema_alias", "timezone_drift", "sentinel_values", "dependency_breakage"],
        },
        "heldout_task3_contract_alias_family": {
            "profile_family": "heldout_referential",
            "novelty_axes": ["unknown_aliases", "contract_extension", "dependency_breakage"],
        },
        "heldout_task3_dependency_rollup_family": {
            "profile_family": "heldout_referential",
            "novelty_axes": ["timezone_drift", "pricing_drift", "dependency_breakage"],
        },
    }[profile]

    if profile == "customer_product_contract_drift":
        customers_broken["age"] = customers_broken["age"].astype(object)
        customers_broken.loc[2, "age"] = None
        products_broken["unit_price"] = products_broken["unit_price"].astype(object)
        products_broken.loc[[0, 1, 3], "unit_price"] = ["$20.00", "$50.00", "$35.00"]
        products_broken.loc[1, "category"] = " IoT "
        orders_broken["customer_id"] = orders_broken["customer_id"].astype(str)
        orders_broken["quantity"] = orders_broken["quantity"].astype(str)
        orders_broken.loc[4, "quantity"] = "5 units"
        orders_broken.loc[7, "quantity"] = "1 units"
        orders_broken.loc[[0, 2, 4, 8], "order_date"] = ["03/01/2026", "03/03/2026", "03/05/2026", "03/09/2026"]
    elif profile == "alias_and_encoding_regression":
        customers_broken["age"] = customers_broken["age"].astype(object)
        customers_broken.loc[1, "email"] = " RAVI@EXAMPLE.COM "
        customers_broken.loc[3, "age"] = None
        products_broken = products_broken.rename(columns={"category": "product_category"})
        products_broken["unit_price"] = products_broken["unit_price"].astype(object)
        products_broken.loc[[1, 2], "unit_price"] = ["$50.00", "$80.00"]
        products_broken.loc[1, "product_category"] = " IoT "
        products_broken = pd.concat([products_broken, products_broken.iloc[[2]]], ignore_index=True)
        orders_broken["customer_id"] = orders_broken["customer_id"].astype(str)
        orders_broken["quantity"] = orders_broken["quantity"].astype(str)
        orders_broken.loc[0, "quantity"] = "2 units"
        orders_broken.loc[2, "order_date"] = "03/03/2026"
        orders_broken.loc[8, "order_date"] = "03/09/2026"
        orders_broken = pd.concat([orders_broken, orders_broken.iloc[[1]]], ignore_index=True)
    elif profile == "sentinel_reference_breakage":
        customers_broken["age"] = customers_broken["age"].astype(object)
        customers_broken.loc[4, "age"] = "unknown"
        customers_broken.loc[6, "age"] = "unknown"
        products_broken["unit_price"] = products_broken["unit_price"].astype(object)
        products_broken.loc[2, "unit_price"] = "999999.0"
        products_broken = pd.concat([products_broken, products_broken.iloc[[0]]], ignore_index=True)
        orders_broken["customer_id"] = orders_broken["customer_id"].astype(str)
        orders_broken["quantity"] = orders_broken["quantity"].astype(str)
        orders_broken.loc[1, "quantity"] = "missing"
        orders_broken.loc[3, "quantity"] = "2 units"
        orders_broken.loc[8, "quantity"] = "missing"
        orders_broken = pd.concat([orders_broken, orders_broken.iloc[[3]]], ignore_index=True)
    elif profile == "timezone_currency_consistency_incident":
        customers_broken["age"] = customers_broken["age"].astype(object)
        customers_broken.loc[0, "age"] = None
        products_broken["unit_price"] = products_broken["unit_price"].astype(object)
        products_broken.loc[[0, 1, 3], "unit_price"] = ["$20.00", "$50.00", "$35.00"]
        products_broken.loc[1, "category"] = " IoT "
        orders_broken["customer_id"] = orders_broken["customer_id"].astype(str)
        orders_broken["quantity"] = orders_broken["quantity"].astype(str)
        orders_broken.loc[[0, 1, 2, 3, 4, 8], "order_date"] = [
            "2026-03-01T00:00:00+05:30",
            "02/03/2026",
            "2026-03-03T00:00:00+05:30",
            "04/03/2026",
            "2026-03-05T00:00:00+05:30",
            "09/03/2026",
        ]
        orders_broken = pd.concat([orders_broken, orders_broken.iloc[[0]]], ignore_index=True)
    elif profile == "heldout_task3_contract_alias_family":
        customers_broken = customers_broken.rename(columns={"email": "email_address"})
        customers_broken["age"] = customers_broken["age"].astype(object)
        customers_broken.loc[[1, 7], "age"] = [None, "unknown"]
        customers_broken.loc[4, "email_address"] = " MEERA@EXAMPLE.COM "
        products_broken = products_broken.rename(columns={"product_name": "item_name", "unit_price": "list_price"})
        products_broken["list_price"] = products_broken["list_price"].astype(object)
        products_broken.loc[[0, 2, 5], "list_price"] = ["$20.00", "$80.00", "$44.00"]
        products_broken = pd.concat([products_broken, products_broken.iloc[[4]]], ignore_index=True)
        orders_broken = orders_broken.rename(columns={"order_date": "event_date"})
        orders_broken["customer_id"] = orders_broken["customer_id"].astype(str)
        orders_broken["product_id"] = orders_broken["product_id"].astype(str)
        orders_broken["quantity"] = orders_broken["quantity"].astype(str)
        orders_broken.loc[[0, 2, 4, 8, 11], "event_date"] = [
            "2026-03-01T00:00:00+05:30",
            "2026-03-03T00:00:00+05:30",
            "2026-03-05T00:00:00+05:30",
            "2026-03-09T00:00:00+05:30",
            "2026-03-12T00:00:00+05:30",
        ]
        orders_broken.loc[[1, 6], "quantity"] = ["1 units", "4 units"]
        orders_broken = pd.concat([orders_broken, orders_broken.iloc[[2]]], ignore_index=True)
    elif profile == "heldout_task3_dependency_rollup_family":
        customers_broken["age"] = customers_broken["age"].astype(object)
        customers_broken.loc[[0, 5], "age"] = [None, "unknown"]
        products_broken["unit_price"] = products_broken["unit_price"].astype(object)
        products_broken.loc[[1, 2, 6], "unit_price"] = ["$50.00", "999999.0", "$91.00"]
        products_broken.loc[3, "category"] = " IoT "
        orders_broken["customer_id"] = orders_broken["customer_id"].astype(str)
        orders_broken["quantity"] = orders_broken["quantity"].astype(str)
        orders_broken.loc[[1, 3, 8, 14], "quantity"] = ["missing", "2 units", "missing", "2 units"]
        orders_broken.loc[[0, 1, 2, 3, 4, 8, 14], "order_date"] = [
            "2026-03-01T00:00:00+05:30",
            "02/03/2026",
            "2026-03-03T00:00:00+05:30",
            "04/03/2026",
            "2026-03-05T00:00:00+05:30",
            "09/03/2026",
            "2026-03-15T00:00:00+05:30",
        ]
        orders_broken.loc[[0, 4, 10], "total_price"] = [999.0, 111.0, 999.0]
        orders_broken = pd.concat([orders_broken, orders_broken.iloc[[3]]], ignore_index=True)
    else:
        customers_broken["age"] = customers_broken["age"].astype(object)
        customers_broken.loc[3, "email"] = " KABIR@EXAMPLE.COM "
        customers_broken.loc[0, "age"] = None
        products_broken = products_broken.rename(columns={"category": "product_category"})
        products_broken["unit_price"] = products_broken["unit_price"].astype(object)
        products_broken.loc[[0, 1, 3], "unit_price"] = ["$20.00", "$50.00", "$35.00"]
        products_broken.loc[2, "unit_price"] = "999999.0"
        products_broken.loc[1, "product_category"] = " IoT "
        products_broken = pd.concat([products_broken, products_broken.iloc[[1]]], ignore_index=True)
        orders_broken["customer_id"] = orders_broken["customer_id"].astype(str)
        orders_broken["quantity"] = orders_broken["quantity"].astype(str)
        orders_broken.loc[1, "quantity"] = "missing"
        orders_broken.loc[3, "quantity"] = "2 units"
        orders_broken.loc[8, "quantity"] = "missing"
        orders_broken.loc[[0, 1, 2, 3, 4, 8], "order_date"] = [
            "2026-03-01T00:00:00+05:30",
            "02/03/2026",
            "2026-03-03T00:00:00+05:30",
            "04/03/2026",
            "2026-03-05T00:00:00+05:30",
            "09/03/2026",
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
            "profile_family": profile_manifest["profile_family"],
            "novelty_axes": profile_manifest["novelty_axes"],
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
