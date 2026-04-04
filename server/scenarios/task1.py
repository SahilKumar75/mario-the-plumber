from __future__ import annotations

import numpy as np
import pandas as pd

from .shared import SYNTHETIC_DATA_NOTES, Scenario, expected_types, patterns_for_profile, sample_profile


def generate_task1(
    rng: np.random.Generator,
    seed: int | None,
    split: str,
) -> Scenario:
    profile = sample_profile(1, split, rng)
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

    if profile in {"currency_contract_regression", "signup_contract_drift"} or rng.random() < 0.4:
        broken["monthly_spend"] = broken["monthly_spend"].astype(object)
        currency_rows = rng.choice(len(broken), size=int(rng.integers(1, 3)), replace=False)
        for row in currency_rows:
            value = ground_truth.loc[row, "monthly_spend"]
            broken.loc[row, "monthly_spend"] = f"${value:,.2f} USD"

    if profile in {"ingestion_date_contract_drift", "signup_contract_drift"} or split == "eval" or rng.random() < 0.5:
        broken["signup_date"] = broken["signup_date"].astype(object)
        date_rows = rng.choice(len(broken), size=int(rng.integers(2, 5)), replace=False)
        for row in date_rows:
            value = pd.to_datetime(ground_truth.loc[row, "signup_date"])
            broken.loc[row, "signup_date"] = value.strftime("%d/%m/%Y")
    if profile == "signup_contract_drift":
        broken = broken.rename(columns={"signup_date": "signup_dt"})

    return Scenario(
        task_id=1,
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
                "ingestion_job_failures": 1,
                "contract_validation_failures": 1 if profile == "signup_contract_drift" else 0,
            },
            "queue_backlog_age_minutes": 0,
        },
    )
