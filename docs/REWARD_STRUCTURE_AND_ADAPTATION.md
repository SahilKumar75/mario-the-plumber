# Reward, Recovery Semantics, and Adaptation

This note describes how Mario scores ETL recovery, how dense shaping relates to true task success, and how held-out generalization is measured in the trace-grounded benchmark.

## Structured Evaluation

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

This keeps the benchmark OpenEnv-compatible while making recovery logic easier to audit.

## Success vs Shaping

Mario always returns a scalar OpenEnv reward, but the intended task success is stricter than one step of score movement.

- **True task success**
  - the repaired table or pipeline clears the task threshold
  - commit happens only after recovery preconditions are satisfied
  - unsafe commit paths should fail even if local table quality temporarily improves
- **Dense shaping**
  - small progress deltas reward better diagnosis and staged repair
  - step cost discourages wandering
  - invalid-action penalties discourage impossible or malformed recovery moves
  - terminal bonus only fires when the commit is actually successful

This means Mario is trying to reward **safe incident recovery**, not just local cleanup.

## Why This Exists

Mario’s harder tasks combine data quality, freshness, dependency repair, backlog recovery, and resource behavior. A single scalar reward is still returned, but these extra signals explain why a score moved.

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

`G(commit -> schema_aligned & backlog_cleared & rollup_consistent & temporal_closure_complete & freshness_sla_met)`

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
- `temporal_backfill`: `0.15`
- `rollup_consistency`: `0.15`
- `temporal_closure`: `0.10`
- `freshness`: `0.15`
- `resource_efficiency`: `0.10`
- `data_quality`: `0.15`

These weights are exposed so benchmark users can see what the environment is optimizing and how hard tasks trade off quality, freshness, backlog clearance, and resource use.

## Exploit Checks

Track A explicitly treats these as invalid benchmark wins:

- deletion-heavy repair that hides missing data instead of restoring it
- premature commit before dependency, backlog, freshness, or rollup recovery is complete
- cosmetic consistency that leaves true downstream semantics broken
- resource overuse that looks active but does not actually recover the incident

## Adaptation Benchmark

Mario now includes a direct one-shot adaptation check:

```bash
python3 scripts/benchmark_adaptation.py --policy-mode heuristic --seeds 1 2 3 4 5 6
```

Current local result:

- train Task 5 mean: `0.9823`
- eval Task 5 mean: `0.7473`
- familiar eval Task 5 mean: `0.9823`
- held-out profile family Task 5 mean: `0.5124`
- held-out family gap: `0.4699`
- held-out profile breakdown:
  - `heldout_temporal_schema_extension_family`: `0.4918`
  - `heldout_temporal_rollup_contract_family`: `0.5372`
  - `heldout_temporal_correction_replay_family`: `0.5081`

Task 5 held-out families now cover distinct temporal novelty axes rather than a single unseen profile, and the temporal commit gate now requires both rollup consistency and closure over the affected replay windows. This makes the adaptation check more discriminative while preserving the same public action contract.
