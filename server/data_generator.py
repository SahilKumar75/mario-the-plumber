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

TASK_THRESHOLDS = {1: 0.85, 2: 0.80, 3: 0.75, 4: 0.78, 5: 0.82}
MAX_STEPS = {1: 10, 2: 15, 3: 25, 4: 30, 5: 35}
TASK_NAMES = {
    1: "Single Table Missing Values",
    2: "Single Table Duplicates and Types",
    3: "Multi-Table Cascading Failure",
    4: "Incremental Pipeline Recovery",
    5: "Temporal ETL Recovery with Reward Machine Structure",
}
TASK_DIFFICULTY = {1: "easy", 2: "medium", 3: "hard", 4: "hard", 5: "hard"}
TASK_TABLES = {
    1: ["single"],
    2: ["single"],
    3: ["orders", "customers", "products"],
    4: ["orders", "products", "daily_summary"],
    5: ["source_orders", "catalog", "hourly_rollup"],
}

SCENARIO_PROFILES: dict[int, dict[str, list[str]]] = {
    1: {
        "train": ["nulls_and_format_drift"],
        "eval": ["nulls_and_date_drift", "currency_format_pressure"],
    },
    2: {
        "train": ["duplicates_and_dtype_drift"],
        "eval": ["duplicates_dtype_and_date_drift", "outlier_and_currency_drift"],
    },
    3: {
        "train": [
            "currency_date_drift",
            "alias_encoding_drift",
            "sentinel_missing_values",
        ],
        "eval": [
            "alias_encoding_drift",
            "timezone_and_currency_drift",
            "sentinel_missing_values",
            "mixed_open_world_breakage",
        ],
    },
    4: {
        "train": [
            "late_batch_resource_pressure",
            "schema_alias_and_units",
            "stale_summary_recovery",
        ],
        "eval": [
            "timezone_alias_burst",
            "schema_alias_and_units",
            "stale_summary_recovery",
            "mixed_operational_open_world",
        ],
    },
    5: {
        "train": [
            "temporal_rollup_recovery",
            "schema_evolution_and_backfill",
            "late_correction_backpressure",
        ],
        "eval": [
            "schema_evolution_and_backfill",
            "late_correction_backpressure",
            "temporal_open_world_shift",
            "heldout_temporal_profile_family",
        ],
    },
}

SYNTHETIC_DATA_NOTES = [
    "Synthetic tables preserve schema-level repair structure, not enterprise-scale row volume.",
    "Utility is benchmarked through relative policy separation across tasks and held-out splits.",
    "Profiles intentionally vary failure combinations so agents cannot rely on a single fixed script.",
]

PROFILE_PATTERNS = {
    "nulls_and_format_drift": ["missing_values", "format_drift"],
    "nulls_and_date_drift": ["missing_values", "date_drift"],
    "currency_format_pressure": ["currency_drift", "format_drift"],
    "duplicates_and_dtype_drift": ["duplicates", "dtype_drift"],
    "duplicates_dtype_and_date_drift": ["duplicates", "dtype_drift", "date_drift"],
    "outlier_and_currency_drift": ["outlier", "currency_drift"],
    "currency_date_drift": ["currency_drift", "date_drift"],
    "alias_encoding_drift": ["schema_alias", "encoding_drift"],
    "sentinel_missing_values": ["sentinel_values", "missing_values"],
    "timezone_and_currency_drift": ["timezone_drift", "currency_drift"],
    "mixed_open_world_breakage": ["schema_alias", "timezone_drift", "sentinel_values"],
    "late_batch_resource_pressure": ["late_batch", "resource_pressure"],
    "schema_alias_and_units": ["schema_alias", "unit_drift"],
    "stale_summary_recovery": ["stale_summary", "downstream_refresh"],
    "timezone_alias_burst": ["timezone_drift", "schema_alias", "workload_burst"],
    "mixed_operational_open_world": ["schema_alias", "timezone_drift", "stale_summary", "resource_pressure"],
    "temporal_rollup_recovery": ["late_batch", "stale_summary", "timestamp_rollup"],
    "schema_evolution_and_backfill": ["schema_alias", "unit_drift", "backfill_required"],
    "late_correction_backpressure": ["late_batch", "resource_pressure", "correction_replay"],
    "temporal_open_world_shift": ["schema_alias", "timezone_drift", "timestamp_rollup", "correction_replay"],
    "heldout_temporal_profile_family": ["schema_alias", "backfill_required", "workload_burst", "timestamp_rollup"],
}

TASK_OBJECTIVE_WEIGHTS: dict[int, dict[str, float]] = {
    3: {"data_quality": 0.55, "dependency_consistency": 0.45},
    4: {
        "data_quality": 0.45,
        "freshness": 0.20,
        "backlog": 0.15,
        "resource_efficiency": 0.10,
        "summary_consistency": 0.10,
    },
    5: {
        "schema_alignment": 0.20,
        "temporal_backfill": 0.20,
        "rollup_consistency": 0.20,
        "freshness": 0.15,
        "resource_efficiency": 0.10,
        "data_quality": 0.15,
    },
}

FORMAL_TASK_SPECS: dict[int, dict[str, object]] = {
    3: {
        "reward_machine_order": [
            "repair_customers",
            "repair_products",
            "repair_orders",
            "restore_dependency_consistency",
            "commit_pipeline",
        ],
        "ltl_hint": "G(commit -> products_clean & customers_clean & orders_clean & dependency_consistent)",
    },
    4: {
        "reward_machine_order": [
            "normalize_orders_stream",
            "scale_resources_if_needed",
            "load_incremental_backlog",
            "refresh_daily_summary",
            "commit_recovery",
        ],
        "ltl_hint": "G(commit -> backlog_cleared & freshness_restored & summary_fresh)",
    },
    5: {
        "reward_machine_order": [
            "reconcile_schema_aliases",
            "repair_catalog_and_source_quality",
            "replay_late_batches",
            "refresh_temporal_rollup",
            "meet_freshness_sla",
            "commit_temporal_pipeline",
        ],
        "ltl_hint": "G(commit -> schema_aligned & backlog_cleared & rollup_consistent & freshness_sla_met)",
    },
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
    if task_id == 5:
        return _generate_task5(rng, seed, split)
    raise ValueError(f"Unsupported task_id: {task_id}")


def benchmark_metadata() -> dict[str, object]:
    """Static benchmark metadata for docs and API endpoints."""

    return {
        "task_names": TASK_NAMES,
        "task_thresholds": TASK_THRESHOLDS,
        "max_steps": MAX_STEPS,
        "scenario_profiles": SCENARIO_PROFILES,
        "synthetic_data_notes": SYNTHETIC_DATA_NOTES,
        "objective_weights": TASK_OBJECTIVE_WEIGHTS,
        "formal_task_specs": FORMAL_TASK_SPECS,
    }


def _sample_profile(
    task_id: int,
    split: str,
    rng: np.random.Generator,
) -> str:
    profiles = SCENARIO_PROFILES.get(task_id, {}).get(split)
    if not profiles:
        profiles = SCENARIO_PROFILES.get(task_id, {}).get("train", ["baseline"])
    return str(rng.choice(profiles))


def _patterns_for_profile(profile: str) -> list[str]:
    return PROFILE_PATTERNS.get(profile, [profile])


def _generate_task1(
    rng: np.random.Generator,
    seed: int | None,
    split: str,
) -> Scenario:
    profile = _sample_profile(1, split, rng)
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

    if profile in {"currency_format_pressure"} or rng.random() < 0.4:
        broken["monthly_spend"] = broken["monthly_spend"].astype(object)
        currency_rows = rng.choice(len(broken), size=int(rng.integers(1, 3)), replace=False)
        for row in currency_rows:
            value = ground_truth.loc[row, "monthly_spend"]
            broken.loc[row, "monthly_spend"] = f"${value:,.2f} USD"

    if profile in {"nulls_and_date_drift"} or split == "eval" or rng.random() < 0.5:
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
        metadata={
            "scenario_profile": profile,
            "open_world_patterns": _patterns_for_profile(profile),
            "synthetic_data_notes": SYNTHETIC_DATA_NOTES,
        },
    )


def _generate_task2(
    rng: np.random.Generator,
    seed: int | None,
    split: str,
) -> Scenario:
    profile = _sample_profile(2, split, rng)
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
    if profile == "outlier_and_currency_drift" or rng.random() < 0.3:
        broken.loc[0, "amount"] = "999999.0"
    if profile in {"duplicates_dtype_and_date_drift", "outlier_and_currency_drift"} or split == "eval" or rng.random() < 0.5:
        broken["amount"] = broken["amount"].map(lambda value: f"INR {value}")
    if profile == "duplicates_dtype_and_date_drift" or split == "eval" or rng.random() < 0.4:
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
        metadata={
            "scenario_profile": profile,
            "open_world_patterns": _patterns_for_profile(profile),
            "synthetic_data_notes": SYNTHETIC_DATA_NOTES,
        },
    )


def _generate_task3(
    rng: np.random.Generator,
    seed: int | None,
    split: str,
) -> Scenario:
    profile = _sample_profile(3, split, rng)
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
    if profile in {"alias_encoding_drift", "mixed_open_world_breakage"} or split == "eval" or rng.random() < 0.5:
        email_row = int(rng.integers(0, len(customers_broken)))
        customers_broken.loc[email_row, "email"] = f" {customers_broken.loc[email_row, 'email'].upper()} "
    if profile in {"sentinel_missing_values", "mixed_open_world_breakage"}:
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
    if profile in {"alias_encoding_drift", "mixed_open_world_breakage"}:
        products_broken = products_broken.rename(columns={"category": "product_category"})
    category_column = "product_category" if "product_category" in products_broken.columns else "category"
    if split == "eval" or profile in {"currency_date_drift", "timezone_and_currency_drift", "mixed_open_world_breakage"} or rng.random() < 0.6:
        products_broken.loc[1, category_column] = " IoT "
    if split == "eval" or profile in {"currency_date_drift", "timezone_and_currency_drift", "mixed_open_world_breakage"} or rng.random() < 0.5:
        formatted_rows = rng.choice(len(products_broken), size=min(2, len(products_broken)), replace=False)
        for row in formatted_rows:
            raw_value = pd.to_numeric(products_broken.loc[row, "unit_price"], errors="coerce")
            if pd.notna(raw_value):
                products_broken.loc[row, "unit_price"] = f"${raw_value:,.2f}"

    orders_broken = orders_truth.copy()
    product_prices = products_truth.set_index("product_id")["unit_price"]
    orders_broken["customer_id"] = orders_broken["customer_id"].astype(str)
    orders_broken["quantity"] = orders_broken["quantity"].astype(str)
    if split == "eval" or profile in {"sentinel_missing_values", "mixed_open_world_breakage"}:
        quantity_rows = rng.choice(len(orders_broken), size=int(rng.integers(1, 3)), replace=False)
        for row in quantity_rows:
            orders_broken.loc[row, "quantity"] = "missing"
    elif rng.random() < 0.5:
        quantity_rows = rng.choice(len(orders_broken), size=int(rng.integers(1, 3)), replace=False)
        for row in quantity_rows:
            orders_broken.loc[row, "quantity"] = f"{orders_truth.loc[row, 'quantity']} units"
    if profile in {"timezone_and_currency_drift", "mixed_open_world_breakage"}:
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
        metadata={
            "scenario_profile": profile,
            "open_world_patterns": _patterns_for_profile(profile),
            "synthetic_data_notes": SYNTHETIC_DATA_NOTES,
        },
    )


def _generate_task4(
    rng: np.random.Generator,
    seed: int | None,
    split: str,
) -> Scenario:
    profile = _sample_profile(4, split, rng)
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
        if profile in {"timezone_alias_burst", "mixed_operational_open_world"} and index == 0:
            orders_broken.iloc[row, orders_broken.columns.get_loc("event_ts")] = ts.strftime("%Y-%m-%d %H:%M:%S+05:30")
        elif index % 2 == 0:
            orders_broken.iloc[row, orders_broken.columns.get_loc("event_ts")] = ts.strftime("%d/%m/%Y %H:%M")
        else:
            orders_broken.iloc[row, orders_broken.columns.get_loc("event_ts")] = ts.strftime("%m-%d-%Y %H:%M")
    if profile in {"schema_alias_and_units", "timezone_alias_burst", "mixed_operational_open_world"}:
        orders_broken = orders_broken.rename(columns={"event_ts": "event_time"})

    if split == "eval" or profile in {"mixed_operational_open_world"} or rng.random() < 0.7:
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
    if split == "eval" or profile in {"schema_alias_and_units", "mixed_operational_open_world"} or rng.random() < 0.6:
        products_broken.iloc[1, products_broken.columns.get_loc("category")] = " IoT "
    if profile in {"schema_alias_and_units", "mixed_operational_open_world"}:
        products_broken = products_broken.rename(columns={"category": "product_segment"})

    summary_broken["event_date"] = summary_broken["event_date"].astype(object)
    if len(summary_broken) > 0:
        first_date = pd.to_datetime(summary_broken.iloc[0]["event_date"])
        summary_broken.iloc[0, summary_broken.columns.get_loc("event_date")] = first_date.strftime("%d/%m/%Y")
    summary_broken["total_revenue"] = (summary_broken["total_revenue"] * 0.92).round(2)
    if profile in {"stale_summary_recovery", "mixed_operational_open_world"}:
        summary_broken = summary_broken.rename(columns={"event_date": "business_date"})

    backlog_rows = len(pending_orders)
    required_resource_level = 2 if backlog_rows <= 2 else 3
    freshness_lag_minutes = 90 if split == "train" else 150
    if profile in {"timezone_alias_burst", "mixed_operational_open_world"}:
        freshness_lag_minutes += 30

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
            "scenario_profile": profile,
            "open_world_patterns": _patterns_for_profile(profile),
            "synthetic_data_notes": SYNTHETIC_DATA_NOTES,
        },
    )


def _generate_task5(
    rng: np.random.Generator,
    seed: int | None,
    split: str,
) -> Scenario:
    profile = _sample_profile(5, split, rng)
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

    if profile in {"schema_evolution_and_backfill", "temporal_open_world_shift", "heldout_temporal_profile_family"}:
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
    if profile in {"schema_evolution_and_backfill", "heldout_temporal_profile_family"}:
        catalog_broken = catalog_broken.rename(columns={"category": "product_segment"})
    category_column = "product_segment" if "product_segment" in catalog_broken.columns else "category"
    catalog_broken.iloc[1, catalog_broken.columns.get_loc(category_column)] = " IoT "
    catalog_broken = pd.concat([catalog_broken, catalog_broken.iloc[[2]]], ignore_index=True)

    if profile in {"temporal_rollup_recovery", "temporal_open_world_shift", "heldout_temporal_profile_family"}:
        rollup_broken = rollup_broken.rename(columns={"hour_bucket": "window_start"})
    rollup_time_column = "window_start" if "window_start" in rollup_broken.columns else "hour_bucket"
    if len(rollup_broken) > 0:
        rollup_broken.iloc[0, rollup_broken.columns.get_loc(rollup_time_column)] = "29/03/2026 10:00"
    rollup_broken["gross_revenue"] = (rollup_broken["gross_revenue"] * 0.88).round(2)

    backlog_rows = len(pending_orders)
    required_resource_level = 3 if backlog_rows >= 4 else 2
    freshness_lag_minutes = 180 if split == "eval" else 120

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
            "source_orders": _expected_types(source_truth),
            "catalog": _expected_types(catalog_truth),
            "hourly_rollup": _expected_types(rollup_truth),
        },
        active_table="source_orders",
        split=split,
        metadata={
            "pending_orders": pending_orders,
            "backlog_rows": backlog_rows,
            "freshness_lag_minutes": freshness_lag_minutes,
            "resource_level": 1,
            "required_resource_level": required_resource_level,
            "pending_batches": 2 if backlog_rows > 0 else 0,
            "downstream_stale": True,
            "workload_pressure": 0.95 if split == "eval" else 0.8,
            "scenario_profile": profile,
            "open_world_patterns": _patterns_for_profile(profile),
            "synthetic_data_notes": SYNTHETIC_DATA_NOTES,
            "heldout_profile_family": profile == "heldout_temporal_profile_family",
            "adaptation_target": "Recover an unseen temporal profile family in one episode.",
            "task_spec": FORMAL_TASK_SPECS[5],
            "tradeoff_weights": TASK_OBJECTIVE_WEIGHTS[5],
        },
    )


def _expected_types(frame: pd.DataFrame) -> dict[str, str]:
    return {column: str(dtype) for column, dtype in frame.dtypes.items()}
