from __future__ import annotations

import numpy as np
import pandas as pd

from .shared import SYNTHETIC_DATA_NOTES, Scenario, expected_types, patterns_for_profile, sample_profile


def generate_task2(
    rng: np.random.Generator,
    seed: int | None,
    split: str,
) -> Scenario:
    profile = sample_profile(2, split, rng)
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
    dtype_cols = ["age", "amount"]
    primary_drift_col, secondary_drift_col = (dtype_cols if rng.random() < 0.5 else list(reversed(dtype_cols)))
    broken[primary_drift_col] = broken[primary_drift_col].astype(str)
    if rng.random() < 0.7:
        broken[secondary_drift_col] = broken[secondary_drift_col].astype(str)
    broken = pd.concat([broken, broken.iloc[[1, 4]]], ignore_index=True)

    if profile == "outlier_currency_regression" or rng.random() < 0.3:
        broken["amount"] = broken["amount"].astype(object)
        broken.loc[0, "amount"] = "999999.0"
    if profile in {"event_contract_breakage", "outlier_currency_regression"} or split == "eval" or rng.random() < 0.5:
        broken["amount"] = broken["amount"].astype(object).map(lambda value: f"INR {value}")

    if profile == "event_contract_breakage" or split == "eval" or rng.random() < 0.4:
        broken["event_date"] = broken["event_date"].astype(object)
        date_rows = rng.choice(len(broken), size=int(rng.integers(2, 5)), replace=False)
        for row in date_rows:
            value = pd.to_datetime(str(ground_truth.loc[min(row, len(ground_truth) - 1), "event_date"]))
            broken.loc[row, "event_date"] = value.strftime("%m-%d-%Y")
    if profile == "event_contract_breakage":
        broken = broken.rename(columns={"event_date": "event_time"})

    return Scenario(
        task_id=2,
        seed=seed,
        broken_tables={"single": broken},
        ground_truth_tables={"single": ground_truth},
        expected_types={"single": expected_types(ground_truth)},
        active_table="single",
        split=split,
        metadata={
            "scenario_profile": profile,
            "open_world_patterns": patterns_for_profile(profile),
            "synthetic_data_notes": SYNTHETIC_DATA_NOTES,
            "recent_failure_counters": {
                "retry_replay_failures": 1,
                "validation_failures": 2 if profile in {"dtype_validation_regression", "event_contract_breakage"} else 1,
            },
            "queue_backlog_age_minutes": 0,
        },
    )
