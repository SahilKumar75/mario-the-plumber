"""Shared benchmark catalog and scenario metadata."""

from __future__ import annotations

BENCHMARK_VERSION = "2.1"

RUNTIME_MODES = {
    "benchmark": {
        "summary": "Default self-contained trace-grounded benchmark episodes for ETL repair and recovery.",
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
    1: "Ingestion Contract Repair",
    2: "Validation and Event Stabilization",
    3: "Referential Repair and Cascading Recovery",
    4: "Incremental ETL Incident Recovery",
    5: "Temporal Rollup Recovery",
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
        "train": ["ingestion_null_burst", "signup_contract_drift"],
        "eval": ["ingestion_date_contract_drift", "currency_contract_regression"],
    },
    2: {
        "train": ["duplicate_event_retry", "dtype_validation_regression"],
        "eval": ["event_contract_breakage", "outlier_currency_regression"],
    },
    3: {
        "train": [
            "customer_product_contract_drift",
            "alias_and_encoding_regression",
            "sentinel_reference_breakage",
        ],
        "eval": [
            "alias_and_encoding_regression",
            "timezone_currency_consistency_incident",
            "sentinel_reference_breakage",
            "cascading_reference_outage",
            "heldout_task3_contract_alias_family",
            "heldout_task3_dependency_rollup_family",
        ],
    },
    4: {
        "train": [
            "late_batch_resource_incident",
            "schema_alias_unit_regression",
            "stale_summary_oncall_recovery",
        ],
        "eval": [
            "schema_alias_unit_regression",
            "stale_summary_oncall_recovery",
            "heldout_task4_batch_shape_family",
            "heldout_task4_summary_contract_family",
            "heldout_task4_worker_pressure_family",
            "heldout_task4_freshness_pressure_family",
        ],
    },
    5: {
        "train": [
            "temporal_rollup_backfill_incident",
            "schema_evolution_backfill_recovery",
            "late_correction_backpressure_incident",
        ],
        "eval": [
            "schema_evolution_backfill_recovery",
            "late_correction_backpressure_incident",
            "heldout_temporal_schema_extension_family",
            "heldout_temporal_rollup_contract_family",
            "heldout_temporal_correction_replay_family",
        ],
    },
}

SYNTHETIC_DATA_NOTES = [
    "Self-contained fixture packs bundle broken tables with DAG-run and warehouse-event traces.",
    "Operational metadata is trace-grounded, but packaged locally so the environment stays reproducible and offline.",
    "Profiles intentionally vary failure combinations so agents cannot rely on a single fixed script.",
]

PROFILE_PATTERNS = {
    "ingestion_null_burst": ["missing_values", "format_drift"],
    "signup_contract_drift": ["missing_values", "schema_alias", "date_drift"],
    "ingestion_date_contract_drift": ["missing_values", "date_drift"],
    "currency_contract_regression": ["currency_drift", "format_drift"],
    "duplicate_event_retry": ["duplicates", "dtype_drift"],
    "dtype_validation_regression": ["duplicates", "dtype_drift", "validation_regression"],
    "event_contract_breakage": ["duplicates", "dtype_drift", "date_drift", "schema_alias"],
    "outlier_currency_regression": ["outlier", "currency_drift", "validation_regression"],
    "customer_product_contract_drift": ["currency_drift", "date_drift", "dependency_breakage"],
    "alias_and_encoding_regression": ["schema_alias", "encoding_drift", "dependency_breakage"],
    "sentinel_reference_breakage": ["sentinel_values", "missing_values", "dependency_breakage"],
    "timezone_currency_consistency_incident": ["timezone_drift", "currency_drift", "dependency_breakage"],
    "cascading_reference_outage": ["schema_alias", "timezone_drift", "sentinel_values", "dependency_breakage"],
    "heldout_task3_contract_alias_family": [
        "schema_alias",
        "encoding_drift",
        "contract_extension",
        "dependency_breakage",
    ],
    "heldout_task3_dependency_rollup_family": [
        "timezone_drift",
        "currency_drift",
        "sentinel_values",
        "dependency_breakage",
    ],
    "late_batch_resource_incident": ["late_batch", "resource_pressure", "retry_pressure"],
    "schema_alias_unit_regression": ["schema_alias", "unit_drift", "validation_regression"],
    "stale_summary_oncall_recovery": ["stale_summary", "downstream_refresh", "unsafe_commit_risk"],
    "timezone_alias_burst_incident": ["timezone_drift", "schema_alias", "workload_burst", "unsafe_commit_risk"],
    "mixed_operational_recovery_incident": ["schema_alias", "timezone_drift", "stale_summary", "resource_pressure", "unsafe_commit_risk"],
    "heldout_task4_batch_shape_family": [
        "late_batch",
        "workload_burst",
        "replay_window_shape_shift",
        "resource_pressure",
        "unsafe_commit_risk",
    ],
    "heldout_task4_summary_contract_family": [
        "schema_alias",
        "summary_contract_drift",
        "stale_summary",
        "unsafe_commit_risk",
    ],
    "heldout_task4_worker_pressure_family": [
        "late_batch",
        "resource_pressure",
        "retry_pressure",
        "stale_summary",
    ],
    "heldout_task4_freshness_pressure_family": [
        "freshness_pressure",
        "stale_summary",
        "workload_burst",
        "downstream_refresh",
    ],
    "temporal_rollup_backfill_incident": ["late_batch", "stale_summary", "timestamp_rollup", "backfill_required"],
    "schema_evolution_backfill_recovery": ["schema_alias", "unit_drift", "backfill_required", "schema_extension"],
    "late_correction_backpressure_incident": ["late_batch", "resource_pressure", "correction_replay", "unsafe_commit_risk"],
    "temporal_open_world_shift_incident": ["schema_alias", "timezone_drift", "timestamp_rollup", "correction_replay"],
    "heldout_temporal_schema_extension_family": [
        "schema_alias",
        "schema_extension",
        "backfill_required",
        "workload_burst",
        "timestamp_rollup",
    ],
    "heldout_temporal_rollup_contract_family": [
        "rollup_contract_drift",
        "schema_alias",
        "backfill_required",
        "timestamp_rollup",
    ],
    "heldout_temporal_correction_replay_family": [
        "late_batch",
        "correction_replay",
        "schema_alias",
        "workload_burst",
        "freshness_pressure",
    ],
}

PROFILE_DESCRIPTIONS = {
    "ingestion_null_burst": "A first-line ingestion batch arrived with missing numeric fields and lightweight formatting noise.",
    "signup_contract_drift": "The source ingestion contract changed lightweight column naming and date formatting for a customer feed.",
    "ingestion_date_contract_drift": "The feed landed with mixed signup-date formats after an upstream ingestion formatter changed.",
    "currency_contract_regression": "Currency values were serialized with business-facing formatting instead of numeric pipeline-safe values.",
    "duplicate_event_retry": "A retry loop replayed already-processed events, creating duplicates alongside dtype drift.",
    "dtype_validation_regression": "A validation regression left duplicate events and stringified numerics in a pre-downstream transaction table.",
    "event_contract_breakage": "The event contract drifted across timestamp and column naming, breaking downstream validation assumptions.",
    "outlier_currency_regression": "A bad event payload introduced an extreme outlier while amount formatting regressed to business-display strings.",
    "customer_product_contract_drift": "Customer, product, and order tables disagree on contracts after upstream schema and pricing drift.",
    "alias_and_encoding_regression": "A source migration introduced alias drift and encoding cleanup issues that cascade into joins and totals.",
    "sentinel_reference_breakage": "Sentinel placeholders leaked into key fields, causing referential repair and dependency recovery work.",
    "timezone_currency_consistency_incident": "Order timestamps and product prices drifted together, breaking downstream consistency checks.",
    "cascading_reference_outage": "Multiple upstream regressions caused cross-table reference breakage and downstream order-total corruption.",
    "heldout_task3_contract_alias_family": "A held-out referential family introduces unfamiliar contract aliases across customer, product, and order tables before totals can be trusted again.",
    "heldout_task3_dependency_rollup_family": "A held-out referential family combines heavier dependency mismatches with timestamp and pricing drift across linked tables.",
    "late_batch_resource_incident": "An incremental batch is delayed while the recovery worker is under-provisioned during an on-call incident.",
    "schema_alias_unit_regression": "Upstream alias and unit drift must be normalized before backlog replay and downstream refresh are safe.",
    "stale_summary_oncall_recovery": "The downstream summary is stale after incremental load failure and needs coordinated recovery before commit.",
    "timezone_alias_burst_incident": "A workload burst arrived with timezone and alias drift, creating an urgent incremental recovery incident.",
    "mixed_operational_recovery_incident": "A realistic on-call recovery with drift, stale downstreams, and resource pressure across the pipeline.",
    "heldout_task4_batch_shape_family": "A held-out recovery family introduces unseen replay-window shapes and larger incremental burst patterns before the summary can be trusted again.",
    "heldout_task4_summary_contract_family": "A held-out recovery family shifts summary and upstream contracts together, requiring recovery before freshness and consistency can close.",
    "heldout_task4_worker_pressure_family": "A held-out recovery family combines backlog replay with stronger worker pressure and a noisier incremental recovery path.",
    "heldout_task4_freshness_pressure_family": "A held-out recovery family couples replay, freshness pressure, and stale downstream state so refresh timing matters.",
    "temporal_rollup_backfill_incident": "Late temporal batches and stale rollups require careful backfill and refresh before commit.",
    "schema_evolution_backfill_recovery": "Schema evolution and backfill pressure must be reconciled before temporal rollups become trustworthy again.",
    "late_correction_backpressure_incident": "Late source corrections are piling up while the recovery path is under backpressure.",
    "temporal_open_world_shift_incident": "Temporal profiles shifted across schema, timestamps, and correction replay patterns.",
    "heldout_temporal_schema_extension_family": "A held-out temporal family adds unseen schema-extension and alias drift while late batches and rollup freshness are both degraded.",
    "heldout_temporal_rollup_contract_family": "A held-out temporal family changes rollup naming and revenue-contract semantics while backlog replay is still required.",
    "heldout_temporal_correction_replay_family": "A held-out temporal family mixes late-correction replay pressure, freshness breaches, and alias drift across source and rollup layers.",
}

TASK_OBJECTIVE_WEIGHTS: dict[int, dict[str, float]] = {
    1: {
        "completeness": 0.20,
        "validity": 0.20,
        "consistency": 0.30,
        "accuracy": 0.30,
    },
    2: {
        "completeness": 0.20,
        "validity": 0.20,
        "consistency": 0.30,
        "accuracy": 0.30,
    },
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
        "temporal_backfill": 0.15,
        "rollup_consistency": 0.15,
        "temporal_closure": 0.10,
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
        "transition_predicates": {
            "repair_customers": "customer schema, null, duplicate, and format issues are cleared",
            "repair_products": "product schema, price-format, and duplicate issues are cleared",
            "repair_orders": "order quantity/date/schema issues are cleared",
            "restore_dependency_consistency": "orders.total_price matches quantity * products.unit_price",
            "commit_pipeline": "all upstream tables are structurally clean and commit succeeds",
        },
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
        "transition_predicates": {
            "normalize_orders_stream": "incremental orders stream is structurally repairable",
            "scale_resources_if_needed": "resource level is high enough for replay pressure",
            "load_incremental_backlog": "late batch is replayed into the live orders table",
            "refresh_daily_summary": "daily summary is recomputed from repaired upstream tables",
            "commit_recovery": "backlog cleared, freshness restored, and safe commit completed",
        },
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
        "transition_predicates": {
            "reconcile_schema_aliases": "schema evolution and alias drift are reconciled across temporal inputs",
            "repair_catalog_and_source_quality": "source and catalog data become structurally trustworthy",
            "replay_late_batches": "late corrections and pending batches are replayed safely",
            "refresh_temporal_rollup": "hourly rollup is rebuilt from repaired temporal state",
            "meet_freshness_sla": "freshness lag is reduced below the temporal SLA",
            "commit_temporal_pipeline": "temporal pipeline is safe to commit without stale downstream state",
        },
    },
}

TASK_CARDS = {
    1: {
        "incident_type": "first-line ingestion repair",
        "objective": "Stabilize a customer-ingestion table before downstream consumers see null-heavy or contract-drifted records.",
        "broken_state": "Missing numeric values, lightweight schema alias drift, currency-style formatting, and inconsistent signup-date serialization.",
        "incident_description": "An upstream ingestion job landed a customer batch with missing values and a light contract regression after a feed formatter changed.",
        "diagnosis_signals": [
            "null counts spike in age and monthly_spend",
            "signup date formats stop matching the ingestion contract",
            "missing_expected_columns or alias hints reveal lightweight source drift",
        ],
        "repair_steps": [
            "restore missing numeric values",
            "rename lightweight aliases back to the contract",
            "normalize formatted numerics and signup dates",
        ],
        "recovery_requirements": [
            "all expected columns are present",
            "dtype and format mismatches are cleared",
            "the single table is safe to hand to downstream jobs",
        ],
        "unsafe_commit_conditions": [
            "missing values remain in required numeric fields",
            "ingestion-contract column aliases are still unresolved",
            "signup_date is still serialized in mixed formats",
        ],
        "success_threshold": TASK_THRESHOLDS[1],
        "failure_conditions": ["commit below threshold", "quality collapse", "step budget exhausted"],
        "threshold_rationale": "Task 1 models first-line ingestion repair, so success requires more than cosmetic cleanup; the table should be trustworthy enough for downstream consumption.",
        "target_policy": "diagnose missing values and light contract drift, repair the table conservatively, then commit only when the ingestion contract is stable.",
        "dense_shaping_notes": [
            "stepwise reward tracks quality improvement",
            "the terminal bonus is only earned when the repaired ingestion table clears the success threshold",
        ],
        "exploit_checks": [
            "deletion-heavy repair should not dominate genuine restoration",
            "format cleanup without restoring missing required values should not count as success",
        ],
        "latent_variation_axes": ["column alias drift", "currency formatting", "signup-date serialization"],
        "key_subgoals": ["fill missing values", "repair ingestion contract drift", "commit a downstream-safe table"],
    },
    2: {
        "incident_type": "validation and event stabilization",
        "objective": "Recover a transaction/events table after retries and validation regressions create duplicates, dtype drift, and malformed event fields.",
        "broken_state": "Duplicate event rows, stringified numerics, validation drift in dates or column names, and optional extreme outliers.",
        "incident_description": "A retry path and weak validation rules produced duplicate transaction events and malformed values before the table reached downstream checks.",
        "diagnosis_signals": [
            "duplicate rows appear in the transaction log",
            "amount and age no longer match the expected numeric types",
            "event-date formatting and event-field naming drift from the contract",
        ],
        "repair_steps": [
            "deduplicate replayed events",
            "restore numeric and date contract correctness",
            "remove or stabilize obvious bad event payloads before commit",
        ],
        "recovery_requirements": [
            "the transaction table can pass validation again",
            "duplicates are removed without losing legitimate rows",
            "event fields are normalized for downstream consumers",
        ],
        "unsafe_commit_conditions": [
            "duplicates remain in the transaction log",
            "numeric fields still fail validation",
            "event timestamps or contract columns remain malformed",
        ],
        "success_threshold": TASK_THRESHOLDS[2],
        "failure_conditions": ["commit below threshold", "quality collapse", "step budget exhausted"],
        "threshold_rationale": "Task 2 represents validation cleanup ahead of downstream use, so the commit bar requires duplicate control plus type stability, not just partial cosmetic repair.",
        "target_policy": "treat this as a broken event-validation incident: isolate duplicate/replayed rows, restore contract-valid fields, then commit once downstream validation would pass.",
        "dense_shaping_notes": [
            "stepwise improvements reward stable deduplication and dtype recovery",
            "terminal success still depends on clearing the full table-quality bar",
        ],
        "exploit_checks": [
            "blind row dropping should not outperform correct event stabilization",
            "outlier removal alone should not hide remaining duplicate or dtype issues",
        ],
        "latent_variation_axes": ["event-date serialization", "validation-regression field names", "currency/outlier payloads"],
        "key_subgoals": ["deduplicate retry artifacts", "repair validation-contract fields", "commit a downstream-safe event table"],
    },
    3: {
        "incident_type": "referential consistency incident",
        "objective": "Restore customer, product, and order tables after upstream regressions break referential consistency and cascade into incorrect downstream order totals.",
        "broken_state": "Cross-table dependency errors, alias drift, encoding regressions, price/date drift, and sentinel placeholders leaking into reference fields.",
        "incident_description": "A multi-source contract regression corrupted customer, product, and order tables together, causing downstream joins and total-price checks to fail.",
        "diagnosis_signals": [
            "dependency alerts show order totals no longer match product prices",
            "missing expected columns and alias hints appear across linked tables",
            "recent errors reveal sentinel values, encoding noise, and inconsistent dates",
        ],
        "repair_steps": [
            "repair customer and product contracts first",
            "stabilize order quantity/date fields using the repaired upstream context",
            "restore dependency consistency before attempting commit",
        ],
        "recovery_requirements": [
            "customer, product, and order tables are structurally clean",
            "order totals align with repaired product pricing",
            "referential recovery is complete before pipeline commit",
        ],
        "unsafe_commit_conditions": [
            "orders.total_price still mismatches repaired product pricing",
            "customer or product contracts remain structurally broken",
            "order-table cleanup happened before upstream reference repair completed",
        ],
        "success_threshold": TASK_THRESHOLDS[3],
        "failure_conditions": ["commit before dependency consistency", "quality collapse", "step budget exhausted"],
        "threshold_rationale": "Task 3 models cascading downstream breakage, so success requires both local table cleanup and restored cross-table consistency before commit.",
        "target_policy": "diagnose the broken dependency chain, repair upstream reference tables first, then fix dependent orders and commit only when downstream consistency checks would pass.",
        "dense_shaping_notes": [
            "table-level quality signals help the agent stage repairs across customers, products, and orders",
            "dependency consistency remains a distinct objective that can block safe commit even after local cleanup improves",
        ],
        "exploit_checks": [
            "premature commit before restoring dependency consistency must fail",
            "cosmetic order cleanup without upstream repair must not receive full credit",
        ],
        "latent_variation_axes": ["alias drift", "encoding regressions", "sentinel values", "timezone/currency consistency"],
        "key_subgoals": list(FORMAL_TASK_SPECS[3]["reward_machine_order"]),
    },
    4: {
        "incident_type": "incremental on-call recovery",
        "objective": "Recover a broken incremental ETL pipeline while backlog, freshness lag, and resource pressure create an on-call-style recovery incident.",
        "broken_state": "Pending incremental batches, stale downstream summary, under-provisioned workers, alias/unit drift, and a workload burst or retry backlog.",
        "incident_description": "An incremental load failed midstream, leaving late batches unprocessed, daily summaries stale, and the recovery worker under pressure.",
        "diagnosis_signals": [
            "backlog rows and pending batches remain non-zero",
            "freshness lag and workload pressure are elevated",
            "orchestration alerts show stale downstream state or insufficient resources",
        ],
        "repair_steps": [
            "normalize the live orders/products state so backlog replay is safe",
            "scale resources or sequence recovery actions appropriately",
            "replay the delayed batch and refresh the downstream summary before commit",
        ],
        "recovery_requirements": [
            "incremental backlog is fully replayed",
            "freshness lag is restored to zero",
            "daily_summary is rebuilt from repaired upstream state",
        ],
        "unsafe_commit_conditions": [
            "pending batches or backlog rows still remain",
            "freshness lag has not been restored",
            "daily_summary is still stale or based on unrepaired upstream tables",
        ],
        "success_threshold": TASK_THRESHOLDS[4],
        "failure_conditions": ["commit before backlog/freshness recovery", "quality collapse", "step budget exhausted"],
        "threshold_rationale": "Task 4 represents a live incremental incident, so success requires safe replay and downstream freshness recovery, not only upstream data cleanup.",
        "target_policy": "treat this as an on-call ETL incident: stabilize the stream, provision enough resources, replay backlog, refresh the summary, and only then commit.",
        "dense_shaping_notes": [
            "quality, freshness, backlog, and resource signals all contribute to recovery progress",
            "terminal success still requires safe pipeline recovery rather than isolated table repair",
        ],
        "exploit_checks": [
            "over-scaling resources without clearing backlog should not fake recovery",
            "refreshing the summary before replaying backlog should not count as safe recovery",
        ],
        "latent_variation_axes": ["resource pressure", "alias/unit drift", "stale summaries", "workload bursts"],
        "key_subgoals": list(FORMAL_TASK_SPECS[4]["reward_machine_order"]),
    },
    5: {
        "incident_type": "temporal rollup recovery",
        "objective": "Restore a temporal ETL pipeline after schema evolution, late corrections, and stale rollups leave the warehouse in an unsafe-to-commit state.",
        "broken_state": "Schema evolution, late corrections, held-out temporal incident profiles, stale hourly rollup output, and backlog pressure under a freshness SLA.",
        "incident_description": "A temporal pipeline absorbed schema evolution and late source corrections without a complete replay, leaving rollups stale and freshness guarantees broken.",
        "diagnosis_signals": [
            "schema drift and alias hints appear in source and catalog tables",
            "late batches and freshness lag remain above the temporal SLA",
            "hourly rollup is stale or inconsistent with repaired source state",
        ],
        "repair_steps": [
            "reconcile schema evolution across source and catalog",
            "repair source quality before replaying late corrections",
            "refresh the temporal rollup and restore SLA compliance before commit",
        ],
        "recovery_requirements": [
            "schema alignment is restored across temporal tables",
            "late corrections are replayed safely",
            "hourly_rollup is rebuilt and freshness SLA is satisfied",
        ],
        "unsafe_commit_conditions": [
            "schema aliases remain unresolved across source and catalog",
            "late corrections or backlog remain unreplayed",
            "hourly_rollup is stale or freshness SLA remains violated",
        ],
        "success_threshold": TASK_THRESHOLDS[5],
        "failure_conditions": ["commit before SLA/rollup recovery", "quality collapse", "step budget exhausted"],
        "threshold_rationale": "Task 5 is the strongest safe-recovery task: success means the temporal pipeline is structurally repaired, replayed, rolled up, and back within SLA before commit.",
        "target_policy": "diagnose schema evolution and late-correction pressure, restore upstream integrity, replay safely, refresh the rollup, meet SLA, then commit.",
        "dense_shaping_notes": [
            "reward-machine progress exposes the temporal recovery sequence without changing the scalar OpenEnv reward",
            "freshness and rollup consistency remain distinct from mere source-table cleanup",
        ],
        "exploit_checks": [
            "refreshing the rollup before replaying late batches should not count as success",
            "resource overscaling without SLA recovery should not fake a safe temporal commit",
        ],
        "latent_variation_axes": ["schema evolution", "late corrections", "held-out temporal incident families", "freshness SLA pressure"],
        "key_subgoals": list(FORMAL_TASK_SPECS[5]["reward_machine_order"]),
    },
}

KNOWN_LIMITATIONS = [
    "Row deletion via drop_nulls is intentionally penalized by row-count-sensitive accuracy, so agents should prefer repair over deletion.",
    "The provided inference baseline is a benchmark policy family, not a learned RL training pipeline.",
    "The environment is trace-grounded and self-contained, but it is still a benchmark abstraction rather than a live warehouse integration.",
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
        "profile_descriptions": PROFILE_DESCRIPTIONS,
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
