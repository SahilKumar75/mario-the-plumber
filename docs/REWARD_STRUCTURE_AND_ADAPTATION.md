# Reward Structure and Adaptation Notes

This note captures the concrete benchmark upgrades motivated by the latest reward-design, single-episode transfer, Reward Machine, and LTL papers.

## What Changed In Mario

Mario now exposes a structured layer on top of its scalar OpenEnv reward for the harder tasks.

For Tasks 3, 4, and 5, observations now include:

- `reward_breakdown`
- `objective_breakdown`
- `tradeoff_weights`
- `subgoal_progress`
- `subgoal_order`
- `active_subgoal`
- `reward_machine_state`
- `adaptation_target`
- `heldout_profile_family`

This keeps the benchmark OpenEnv-compatible while making the reward story much more auditable.

## Why This Matters

The papers consistently point to four benchmark design needs:

1. reward should be interpretable, not only scalar
2. multi-objective tradeoffs should be visible
3. harder tasks benefit from explicit temporal / compositional structure
4. adaptation should be tested on unseen profile families, not just seen distributions

Mario now addresses each one:

- reward decomposition: via `reward_breakdown`
- objective visibility: via `objective_breakdown` and `tradeoff_weights`
- formal task structure: via Reward-Machine-style `subgoal_order` and `reward_machine_state`
- adaptation benchmark: via held-out Task 5 profile families and `scripts/benchmark_adaptation.py`

## Task 3 Formal Spec

Subgoal order:

1. `repair_customers`
2. `repair_products`
3. `repair_orders`
4. `restore_dependency_consistency`
5. `commit_pipeline`

LTL-style hint:

`G(commit -> products_clean & customers_clean & orders_clean & dependency_consistent)`

## Task 4 Formal Spec

Subgoal order:

1. `normalize_orders_stream`
2. `scale_resources_if_needed`
3. `load_incremental_backlog`
4. `refresh_daily_summary`
5. `commit_recovery`

LTL-style hint:

`G(commit -> backlog_cleared & freshness_restored & summary_fresh)`

## Task 5 Formal Spec

Task 5 is the new temporal/compositional benchmark task inspired by Reward Machines and temporal-logic task structure.

Tables:

- `source_orders`
- `catalog`
- `hourly_rollup`

Subgoal order:

1. `reconcile_schema_aliases`
2. `repair_catalog_and_source_quality`
3. `replay_late_batches`
4. `refresh_temporal_rollup`
5. `meet_freshness_sla`
6. `commit_temporal_pipeline`

LTL-style hint:

`G(commit -> schema_aligned & backlog_cleared & rollup_consistent & freshness_sla_met)`

## Tradeoff Weights

Task 3:

- `data_quality`: `0.55`
- `dependency_consistency`: `0.45`

Task 4:

- `data_quality`: `0.45`
- `freshness`: `0.20`
- `backlog`: `0.15`
- `resource_efficiency`: `0.10`
- `summary_consistency`: `0.10`

Task 5:

- `schema_alignment`: `0.20`
- `temporal_backfill`: `0.20`
- `rollup_consistency`: `0.20`
- `freshness`: `0.15`
- `resource_efficiency`: `0.10`
- `data_quality`: `0.15`

These are intentionally exposed so benchmark users can understand what the environment is rewarding.

## Adaptation Benchmark

Mario now includes a direct one-shot adaptation check:

```bash
python3 scripts/benchmark_adaptation.py --policy-mode heuristic --seeds 1 2 3 4 5 6
```

Current local result:

- train Task 5 mean: `0.9774`
- eval Task 5 mean: `0.9774`
- held-out profile family Task 5 mean: `0.9767`

This is not yet a full learned-policy transfer benchmark, but it is a much cleaner test of generalization to unseen profile families than the earlier single-seed submission story.

## What This Enables Next

- compare pure-LLM vs heuristic vs hybrid on formal subgoal tasks
- audit which objectives are dominating the reward in harder tasks
- add profile-family-specific reporting
- evolve Task 5 toward a stronger temporal benchmark with in-episode workload shifts
