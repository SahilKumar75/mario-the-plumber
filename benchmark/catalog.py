"""Shared benchmark catalog and scenario metadata."""

from __future__ import annotations

BENCHMARK_VERSION = "2.0"

RUNTIME_MODES = {
    "benchmark": {
        "summary": "Default synthetic benchmark episodes for ETL repair and recovery.",
        "use_case": "Scoring, automated evaluation, and baseline comparison.",
    },
    "incident": {
        "summary": "Same benchmark tasks framed as ETL incident-response sessions.",
        "use_case": "Human debugging, manual inspection, and benchmark demos.",
    },
    "hybrid": {
        "summary": "Benchmark mode plus richer reporting and profile inspection.",
        "use_case": "Live Space demo, review, and benchmark visualization.",
    },
}

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

TASK_CARDS = {
    1: {
        "objective": "Repair a single customer table with nulls and lightweight format drift.",
        "broken_state": "Missing numeric values, optional currency formatting, optional signup-date drift.",
        "success_threshold": TASK_THRESHOLDS[1],
        "failure_conditions": ["commit below threshold", "quality collapse", "step budget exhausted"],
        "key_subgoals": ["fill missing values", "normalize lightweight drift", "commit clean table"],
    },
    2: {
        "objective": "Resolve duplicates, dtype drift, and light outlier/format issues in a transaction table.",
        "broken_state": "Duplicate rows, wrong numeric dtypes, optional event-date drift or outliers.",
        "success_threshold": TASK_THRESHOLDS[2],
        "failure_conditions": ["commit below threshold", "quality collapse", "step budget exhausted"],
        "key_subgoals": ["deduplicate", "repair dtypes", "normalize date/format drift", "commit"],
    },
    3: {
        "objective": "Repair cascading multi-table corruption across customers, products, and orders.",
        "broken_state": "Cross-table dependency errors, alias drift, encoding drift, currency/date drift, sentinel nulls.",
        "success_threshold": TASK_THRESHOLDS[3],
        "failure_conditions": ["commit before dependency consistency", "quality collapse", "step budget exhausted"],
        "key_subgoals": list(FORMAL_TASK_SPECS[3]["reward_machine_order"]),
    },
    4: {
        "objective": "Recover an incremental ETL pipeline under backlog, freshness, and resource pressure.",
        "broken_state": "Pending batches, stale summary, insufficient resources, alias/unit drift, workload burst.",
        "success_threshold": TASK_THRESHOLDS[4],
        "failure_conditions": ["commit before backlog/freshness recovery", "quality collapse", "step budget exhausted"],
        "key_subgoals": list(FORMAL_TASK_SPECS[4]["reward_machine_order"]),
    },
    5: {
        "objective": "Restore a temporal ETL pipeline with schema evolution, backfill, and rollup consistency constraints.",
        "broken_state": "Schema evolution, late corrections, held-out temporal profiles, stale hourly rollup, backlog pressure.",
        "success_threshold": TASK_THRESHOLDS[5],
        "failure_conditions": ["commit before SLA/rollup recovery", "quality collapse", "step budget exhausted"],
        "key_subgoals": list(FORMAL_TASK_SPECS[5]["reward_machine_order"]),
    },
}

KNOWN_LIMITATIONS = [
    "Row deletion via drop_nulls is intentionally penalized by row-count-sensitive accuracy, so agents should prefer repair over deletion.",
    "The provided inference baseline is a benchmark policy family, not a learned RL training pipeline.",
    "Synthetic tables optimize for benchmark utility and policy separation, not enterprise-scale realism.",
]


def benchmark_metadata() -> dict[str, object]:
    """Static benchmark metadata for docs and API endpoints."""

    return {
        "benchmark_version": BENCHMARK_VERSION,
        "task_names": TASK_NAMES,
        "task_thresholds": TASK_THRESHOLDS,
        "max_steps": MAX_STEPS,
        "task_cards": TASK_CARDS,
        "scenario_profiles": SCENARIO_PROFILES,
        "synthetic_data_notes": SYNTHETIC_DATA_NOTES,
        "objective_weights": TASK_OBJECTIVE_WEIGHTS,
        "formal_task_specs": FORMAL_TASK_SPECS,
        "runtime_modes": RUNTIME_MODES,
        "known_limitations": KNOWN_LIMITATIONS,
    }


def sample_profile(
    task_id: int,
    split: str,
    rng,
) -> str:
    """Sample a profile family for the given task/split."""

    profiles = SCENARIO_PROFILES.get(task_id, {}).get(split)
    if not profiles:
        profiles = SCENARIO_PROFILES.get(task_id, {}).get("train", ["baseline"])
    return str(rng.choice(profiles))


def patterns_for_profile(profile: str) -> list[str]:
    """Return the open-world pattern tags for a profile."""

    return PROFILE_PATTERNS.get(profile, [profile])
