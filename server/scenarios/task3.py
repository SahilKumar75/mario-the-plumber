from __future__ import annotations

import numpy as np
import pandas as pd

from .shared import SYNTHETIC_DATA_NOTES, Scenario, expected_types, patterns_for_profile, sample_profile


def generate_task3(
    rng: np.random.Generator,
    seed: int | None,
    split: str,
) -> Scenario:
    profile = sample_profile(3, split, rng)
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
    customers_broken["age"] = customers_broken["age"].astype(object)
    customers_broken.loc[int(rng.integers(0, len(customers_broken))), "age"] = np.nan
    if profile in {"alias_and_encoding_regression", "cascading_reference_outage"} or split == "eval" or rng.random() < 0.5:
        email_row = int(rng.integers(0, len(customers_broken)))
        customers_broken.loc[email_row, "email"] = f" {customers_broken.loc[email_row, 'email'].upper()} "
    if profile in {"sentinel_reference_breakage", "cascading_reference_outage"}:
        sentinel_row = int(rng.integers(0, len(customers_broken)))
        customers_broken.loc[sentinel_row, "age"] = "unknown"

    products_broken = products_truth.copy()
    products_broken["unit_price"] = products_broken["unit_price"].astype(str)
    duplicate_product_row = int(rng.integers(0, len(products_broken)))
    products_broken = pd.concat(
        [products_broken, products_broken.iloc[[duplicate_product_row]]],
        ignore_index=True,
    )
    if rng.random() < 0.5:
        products_broken.loc[0, "unit_price"] = "999999.0"
    if profile in {"alias_and_encoding_regression", "cascading_reference_outage"}:
        products_broken = products_broken.rename(columns={"category": "product_category"})
    category_column = "product_category" if "product_category" in products_broken.columns else "category"
    if split == "eval" or profile in {"customer_product_contract_drift", "timezone_currency_consistency_incident", "cascading_reference_outage"} or rng.random() < 0.6:
        products_broken.loc[1, category_column] = " IoT "
    if split == "eval" or profile in {"customer_product_contract_drift", "timezone_currency_consistency_incident", "cascading_reference_outage"} or rng.random() < 0.5:
        formatted_rows = rng.choice(len(products_broken), size=min(2, len(products_broken)), replace=False)
        for row in formatted_rows:
            raw_value = pd.to_numeric(products_broken.loc[row, "unit_price"], errors="coerce")
            if pd.notna(raw_value):
                products_broken.loc[row, "unit_price"] = f"${raw_value:,.2f}"

    orders_broken = orders_truth.copy()
    orders_broken["customer_id"] = orders_broken["customer_id"].astype(str)
    orders_broken["quantity"] = orders_broken["quantity"].astype(str)
    if split == "eval" or profile in {"sentinel_reference_breakage", "cascading_reference_outage"}:
        quantity_rows = rng.choice(len(orders_broken), size=int(rng.integers(1, 3)), replace=False)
        for row in quantity_rows:
            orders_broken.loc[row, "quantity"] = "missing"
    elif rng.random() < 0.5:
        quantity_rows = rng.choice(len(orders_broken), size=int(rng.integers(1, 3)), replace=False)
        for row in quantity_rows:
            orders_broken.loc[row, "quantity"] = f"{orders_truth.loc[row, 'quantity']} units"
    if profile in {"timezone_currency_consistency_incident", "cascading_reference_outage"}:
        orders_broken["order_date"] = orders_broken["order_date"].astype(object)
        date_rows = rng.choice(len(orders_broken), size=int(rng.integers(2, 5)), replace=False)
        for row in date_rows:
            value = pd.to_datetime(orders_truth.loc[row, "order_date"])
            if row % 2 == 0:
                orders_broken.loc[row, "order_date"] = value.strftime("%Y-%m-%dT00:00:00+05:30")
            else:
                orders_broken.loc[row, "order_date"] = value.strftime("%d/%m/%Y")
    elif split == "eval" or rng.random() < 0.6:
        orders_broken["order_date"] = orders_broken["order_date"].astype(object)
        date_rows = rng.choice(len(orders_broken), size=int(rng.integers(2, 5)), replace=False)
        for row in date_rows:
            value = pd.to_datetime(orders_truth.loc[row, "order_date"])
            if row % 2 == 0:
                orders_broken.loc[row, "order_date"] = value.strftime("%d/%m/%Y")
            else:
                orders_broken.loc[row, "order_date"] = value.strftime("%m-%d-%Y")
    orders_broken["total_price"] = orders_truth["total_price"].astype(float)
    duplicate_order_row = int(rng.integers(0, len(orders_broken)))
    orders_broken = pd.concat([orders_broken, orders_broken.iloc[[duplicate_order_row]]], ignore_index=True)

    return Scenario(
        task_id=3,
        seed=seed,
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
        expected_types={
            "orders": expected_types(orders_truth),
            "customers": expected_types(customers_truth),
            "products": expected_types(products_truth),
        },
        active_table="orders",
        split=split,
        metadata={
            "scenario_profile": profile,
            "open_world_patterns": patterns_for_profile(profile),
            "synthetic_data_notes": SYNTHETIC_DATA_NOTES,
            "recent_failure_counters": {
                "join_validation_failures": 1,
                "downstream_total_mismatches": int(len(orders_truth)),
            },
            "queue_backlog_age_minutes": 0,
        },
    )
