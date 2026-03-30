# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Synthetic scenario generation for PipelineDoctor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TASK_THRESHOLDS = {1: 0.85, 2: 0.80, 3: 0.75}
MAX_STEPS = {1: 10, 2: 15, 3: 25}
TASK_NAMES = {
    1: "Single Table Missing Values",
    2: "Single Table Duplicates and Types",
    3: "Multi-Table Cascading Failure",
}
TASK_DIFFICULTY = {1: "easy", 2: "medium", 3: "hard"}
TASK_TABLES = {
    1: ["single"],
    2: ["single"],
    3: ["orders", "customers", "products"],
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


def generate_scenario(task_id: int, seed: int | None = None) -> Scenario:
    """Build a synthetic broken pipeline for the given task."""

    rng = np.random.default_rng(seed)
    if task_id == 1:
        return _generate_task1(rng, seed)
    if task_id == 2:
        return _generate_task2(rng, seed)
    if task_id == 3:
        return _generate_task3(rng, seed)
    raise ValueError(f"Unsupported task_id: {task_id}")


def _generate_task1(rng: np.random.Generator, seed: int | None) -> Scenario:
    ground_truth = pd.DataFrame(
        {
            "customer_id": [101, 102, 103, 104, 105, 106, 107, 108],
            "age": [22, 31, 45, 28, 36, 41, 25, 33],
            "monthly_spend": [120.0, 240.0, 180.0, 210.0, 160.0, 300.0, 145.0, 190.0],
            "city": ["Delhi", "Noida", "Delhi", "Pune", "Mumbai", "Delhi", "Pune", "Mumbai"],
        }
    )
    broken = ground_truth.copy()
    broken.loc[[1, 4], "age"] = np.nan
    broken.loc[[2], "monthly_spend"] = np.nan
    if rng.random() < 0.2:
        broken.loc[6, "monthly_spend"] = 999999.0

    return Scenario(
        task_id=1,
        seed=seed,
        broken_tables={"single": broken},
        ground_truth_tables={"single": ground_truth},
        expected_types={"single": _expected_types(ground_truth)},
        active_table="single",
    )


def _generate_task2(rng: np.random.Generator, seed: int | None) -> Scenario:
    ground_truth = pd.DataFrame(
        {
            "transaction_id": [1001, 1002, 1003, 1004, 1005, 1006],
            "age": [29, 41, 35, 27, 52, 38],
            "amount": [199.5, 329.0, 149.0, 450.25, 510.0, 275.5],
            "status": ["paid", "paid", "failed", "paid", "failed", "paid"],
        }
    )
    broken = ground_truth.copy()
    broken["age"] = broken["age"].astype(str)
    broken["amount"] = broken["amount"].astype(str)
    broken = pd.concat([broken, broken.iloc[[1, 4]]], ignore_index=True)
    if rng.random() < 0.3:
        broken.loc[0, "amount"] = "999999.0"

    return Scenario(
        task_id=2,
        seed=seed,
        broken_tables={"single": broken},
        ground_truth_tables={"single": ground_truth},
        expected_types={"single": _expected_types(ground_truth)},
        active_table="single",
    )


def _generate_task3(rng: np.random.Generator, seed: int | None) -> Scenario:
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
    customers_broken.loc[2, "age"] = np.nan

    products_broken = products_truth.copy()
    products_broken = pd.concat([products_broken, products_broken.iloc[[1]]], ignore_index=True)
    if rng.random() < 0.5:
        products_broken.loc[0, "unit_price"] = 999999.0

    orders_broken = orders_truth.copy()
    product_prices = products_truth.set_index("product_id")["unit_price"]
    orders_broken["quantity"] = orders_broken["quantity"].astype(str)
    numeric_quantity = pd.to_numeric(orders_truth["quantity"], errors="coerce")
    mapped_price = orders_broken["product_id"].map(product_prices)
    orders_broken["total_price"] = (numeric_quantity + mapped_price).round(2)
    orders_broken = pd.concat([orders_broken, orders_broken.iloc[[2]]], ignore_index=True)

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
    )


def _expected_types(frame: pd.DataFrame) -> dict[str, str]:
    return {column: str(dtype) for column, dtype in frame.dtypes.items()}
