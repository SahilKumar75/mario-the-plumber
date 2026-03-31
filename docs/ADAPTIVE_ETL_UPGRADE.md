# Adaptive ETL Upgrade Notes

This note records the paper-inspired benchmark upgrades added after the initial Mario the Plumber submission pass.

## What Was Added

The benchmark now models more than static data repair:

- held-out `train` and `eval` splits
- stronger schema-drift scenarios
- Task 4 for incremental ETL recovery
- orchestration-aware actions
- workload and freshness signals in the observation
- benchmark reporting and static visuals

## New Task 4

Task 4 is called **Incremental Pipeline Recovery**.

It simulates a live ETL system where:

- the latest batch has not fully landed
- upstream schemas drift in timestamp and numeric formatting
- downstream daily aggregates are stale
- the agent must allocate resources, ingest the delayed batch, refresh the downstream table, and only then commit

### Task 4-specific actions

- `16`: `scale_resources_up`
- `17`: `scale_resources_down`
- `18`: `prioritize_incremental_batch`
- `19`: `refresh_downstream_summary`

### Task 4-specific observation signals

- `schema_drift_count`
- `backlog_rows`
- `freshness_lag_minutes`
- `resource_level`
- `required_resource_level`
- `workload_pressure`
- `pending_batches`
- `downstream_stale`
- `orchestration_alerts`

## Why These Changes Matter

The new papers around ETL automation and RL all point to the same idea:

- ETL becomes an RL problem when it is adaptive, sequential, and operational
- realistic benchmarks should include dynamic load, delayed batches, schema drift, and orchestration choices
- benchmark credibility improves when random, strict model-only, and structured policies separate clearly

Task 4 is the first Mario task that directly reflects those ideas.

## Current Benchmark Shape

Mario now spans four levels:

1. local table repair
2. duplicate and dtype repair
3. cross-table dependency repair
4. online incremental pipeline recovery

This moves the project from “data cleanup benchmark” toward “small ETL operations benchmark.”

## Concrete Next Steps For Mario

These are the highest-value follow-ups after the current upgrade.

1. Add a stricter cost model for Task 4.
Current resource actions change score indirectly. A future version should add an explicit overprovisioning penalty.

2. Add workload bursts over time.
Right now workload pressure is episode metadata. A stronger version would vary it during the episode.

3. Add true schema-drift actions.
Examples:
- reconcile renamed fields
- map new enum values
- upgrade downstream schema safely

4. Add a learned policy baseline.
The repo currently includes random, heuristic, hybrid, and strict pure-LLM evaluation. A learned policy would make the benchmark much more research-like.

5. Add benchmark result snapshots to the repo.
Publish train/eval result JSON or CSV files so charts can be regenerated from tracked artifacts rather than manual values.
