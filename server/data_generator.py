# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Synthetic scenario generation for PipelineDoctor."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TASK_THRESHOLDS = {1: 0.85, 2: 0.80, 3: 0.75, 4: 0.78}
MAX_STEPS = {1: 10, 2: 15, 3: 25, 4: 30}
TASK_NAMES = {
    1: "Single Table Missing Values",
    2: "Single Table Duplicates and Types",
    3: "Multi-Table Cascading Failure",
    4: "Incremental Pipeline Recovery",
}
TASK_DIFFICULTY = {1: "easy", 2: "medium", 3: "hard", 4: "hard"}
TASK_TABLES = {
    1: ["single"],
    2: ["single"],
    3: ["orders", "customers", "products"],
    4: ["orders", "products", "daily_summary"],
}


@dataclass(slots=True)
class Scenario:
    """Single deterministic environment scenario."""

    task_id: int
    seed: int | None
    broken_tables: dict[str, pd.DataFrame]
    ground_truth_tables: dict[str, pd.DataFrame]
    expected_types: dict[str, dict[str, str]]
    active_table: str
    split: str = "train"
    metadata: dict[str, object] = field(default_factory=dict)


def generate_scenario(
    task_id: int,
    seed: int | None = None,
    split: str = "train",
) -> Scenario:
    """Build a synthetic broken pipeline for the given task."""

    rng = np.random.default_rng(seed)
    if task_id == 1:
        return _generate_task1(rng, seed, split)
    if task_id == 2:
        return _generate_task2(rng, seed, split)
    if task_id == 3:
        return _generate_task3(rng, seed, split)
    if task_id == 4:
        return _generate_task4(rng, seed, split)
    raise ValueError(f"Unsupported task_id: {task_id}")


def _generate_task1(
    rng: np.random.Generator,
    seed: int | None,
    split: str,
) -> Scenario:
    ground_truth = pd.DataFrame(
        {
            "customer_id": [101, 102, 103, 104, 105, 106, 107, 108],
            "age": [22, 31, 45, 28, 36, 41, 25, 33],
            "monthly_spend": [120.0, 240.0, 180.0, 210.0, 160.0, 300.0, 145.0, 190.0],
            "city": ["Delhi", "Noida", "Delhi", "Pune", "Mumbai", "Delhi", "Pune", "Mumbai"],
            "signup_date": [
                "2026-01-03",
                "2026-01-11",
                "2026-01-18",
                "2026-02-02",
                "2026-02-10",
                "2026-02-17",
                "2026-03-01",
                "2026-03-08",
            ],
        }
    )
    broken = ground_truth.copy()
    age_null_count = int(rng.integers(1, 3))
    spend_null_count = 1
    age_null_rows = rng.choice(len(broken), size=age_null_count, replace=False)
    spend_candidates = [index for index in range(len(broken)) if index not in age_null_rows]
    spend_null_rows = rng.choice(
        spend_candidates,
        size=min(spend_null_count, len(spend_candidates)),
        replace=False,
    )
    broken.loc[age_null_rows, "age"] = np.nan
    broken.loc[spend_null_rows, "monthly_spend"] = np.nan

    if rng.random() < 0.4:
        broken["monthly_spend"] = broken["monthly_spend"].astype(object)
        currency_rows = rng.choice(len(broken), size=int(rng.integers(1, 3)), replace=False)
        for row in currency_rows:
            value = ground_truth.loc[row, "monthly_spend"]
            broken.loc[row, "monthly_spend"] = f"${value:,.2f} USD"

    if split == "eval" or rng.random() < 0.5:
        broken["signup_date"] = broken["signup_date"].astype(object)
        date_rows = rng.choice(len(broken), size=int(rng.integers(2, 5)), replace=False)
        for row in date_rows:
            value = pd.to_datetime(ground_truth.loc[row, "signup_date"])
            broken.loc[row, "signup_date"] = value.strftime("%d/%m/%Y")

    return Scenario(
        task_id=1,
        seed=seed,
        broken_tables={"single": broken},
        ground_truth_tables={"single": ground_truth},
        expected_types={"single": _expected_types(ground_truth)},
        active_table="single",
        split=split,
    )


def _generate_task2(
    rng: np.random.Generator,
    seed: int | None,
    split: str,
) -> Scenario:
    ground_truth = pd.DataFrame(
        {
            "transaction_id": [1001, 1002, 1003, 1004, 1005, 1006],
            "age": [29, 41, 35, 27, 52, 38],
            "amount": [199.5, 329.0, 149.0, 450.25, 510.0, 275.5],
            "status": ["paid", "paid", "failed", "paid", "failed", "paid"],
            "event_date": [
                "2026-03-01",
                "2026-03-02",
                "2026-03-03",
                "2026-03-04",
                "2026-03-05",
                "2026-03-06",
            ],
        }
    )
    broken = ground_truth.copy()
    broken["age"] = broken["age"].astype(str)
    broken["amount"] = broken["amount"].astype(str)
    broken = pd.concat([broken, broken.iloc[[1, 4]]], ignore_index=True)
    if rng.random() < 0.3:
        broken.loc[0, "amount"] = "999999.0"
    if split == "eval" or rng.random() < 0.5:
        broken["amount"] = broken["amount"].map(lambda value: f"INR {value}")
    if split == "eval" or rng.random() < 0.4:
        broken["event_date"] = broken["event_date"].astype(object)
        date_rows = rng.choice(len(broken), size=int(rng.integers(2, 5)), replace=False)
        for row in date_rows:
            value = pd.to_datetime(str(ground_truth.loc[min(row, len(ground_truth) - 1), "event_date"]))
            broken.loc[row, "event_date"] = value.strftime("%m-%d-%Y")

    return Scenario(
        task_id=2,
        seed=seed,
        broken_tables={"single": broken},
        ground_truth_tables={"single": ground_truth},
        expected_types={"single": _expected_types(ground_truth)},
        active_table="single",
        split=split,
    )


def _generate_task3(
    rng: np.random.Generator,
    seed: int | None,
    split: str,
) -> Scenario:
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
    if split == "eval" or rng.random() < 0.5:
        email_row = int(rng.integers(0, len(customers_broken)))
        customers_broken.loc[email_row, "email"] = f" {customers_broken.loc[email_row, 'email'].upper()} "

    products_broken = products_truth.copy()
    products_broken["unit_price"] = products_broken["unit_price"].astype(str)
    duplicate_product_row = int(rng.integers(0, len(products_broken)))
    products_broken = pd.concat(
        [products_broken, products_broken.iloc[[duplicate_product_row]]],
        ignore_index=True,
    )
    if rng.random() < 0.5:
        products_broken.loc[0, "unit_price"] = "999999.0"
    if split == "eval" or rng.random() < 0.6:
        products_broken.loc[1, "category"] = " IoT "
    if split == "eval" or rng.random() < 0.5:
        formatted_rows = rng.choice(len(products_broken), size=min(2, len(products_broken)), replace=False)
        for row in formatted_rows:
            raw_value = pd.to_numeric(products_broken.loc[row, "unit_price"], errors="coerce")
            if pd.notna(raw_value):
                products_broken.loc[row, "unit_price"] = f"${raw_value:,.2f}"

    orders_broken = orders_truth.copy()
    product_prices = products_truth.set_index("product_id")["unit_price"]
    orders_broken["customer_id"] = orders_broken["customer_id"].astype(str)
    orders_broken["quantity"] = orders_broken["quantity"].astype(str)
    if split == "eval" or rng.random() < 0.5:
        quantity_rows = rng.choice(len(orders_broken), size=int(rng.integers(1, 3)), replace=False)
        for row in quantity_rows:
            orders_broken.loc[row, "quantity"] = f"{orders_truth.loc[row, 'quantity']} units"
    if split == "eval" or rng.random() < 0.6:
        orders_broken["order_date"] = orders_broken["order_date"].astype(object)
        date_rows = rng.choice(len(orders_broken), size=int(rng.integers(2, 5)), replace=False)
        for row in date_rows:
            value = pd.to_datetime(orders_truth.loc[row, "order_date"])
            if row % 2 == 0:
                orders_broken.loc[row, "order_date"] = value.strftime("%d/%m/%Y")
            else:
                orders_broken.loc[row, "order_date"] = value.strftime("%m-%d-%Y")
    numeric_quantity = pd.to_numeric(orders_truth["quantity"], errors="coerce")
    mapped_price = orders_broken["product_id"].map(product_prices)
    orders_broken["total_price"] = (numeric_quantity + mapped_price).round(2)
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
            "orders": _expected_types(orders_truth),
            "customers": _expected_types(customers_truth),
            "products": _expected_types(products_truth),
        },
        active_table="orders",
        split=split,
    )


def _generate_task4(
    rng: np.random.Generator,
    seed: int | None,
    split: str,
) -> Scenario:
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
        if index % 2 == 0:
            orders_broken.iloc[row, orders_broken.columns.get_loc("event_ts")] = ts.strftime("%d/%m/%Y %H:%M")
        else:
            orders_broken.iloc[row, orders_broken.columns.get_loc("event_ts")] = ts.strftime("%m-%d-%Y %H:%M")

    if split == "eval" or rng.random() < 0.7:
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
    if split == "eval" or rng.random() < 0.6:
        products_broken.iloc[1, products_broken.columns.get_loc("category")] = " IoT "

    summary_broken["event_date"] = summary_broken["event_date"].astype(object)
    if len(summary_broken) > 0:
        first_date = pd.to_datetime(summary_broken.iloc[0]["event_date"])
        summary_broken.iloc[0, summary_broken.columns.get_loc("event_date")] = first_date.strftime("%d/%m/%Y")
    summary_broken["total_revenue"] = (summary_broken["total_revenue"] * 0.92).round(2)

    backlog_rows = len(pending_orders)
    required_resource_level = 2 if backlog_rows <= 2 else 3
    freshness_lag_minutes = 90 if split == "train" else 150

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
            "orders": _expected_types(orders_truth),
            "products": _expected_types(products_truth),
            "daily_summary": _expected_types(summary_truth),
        },
        active_table="orders",
        split=split,
        metadata={
            "pending_orders": pending_orders,
            "backlog_rows": backlog_rows,
            "freshness_lag_minutes": freshness_lag_minutes,
            "resource_level": 1,
            "required_resource_level": required_resource_level,
            "pending_batches": 1 if backlog_rows > 0 else 0,
            "downstream_stale": True,
            "workload_pressure": 0.9 if split == "eval" else 0.75,
        },
    )


def _expected_types(frame: pd.DataFrame) -> dict[str, str]:
    return {column: str(dtype) for column, dtype in frame.dtypes.items()}
