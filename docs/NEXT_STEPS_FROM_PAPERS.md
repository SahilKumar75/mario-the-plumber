# What The Papers Imply We Should Build Next

This is the concrete build plan suggested by the latest papers on open-world ML, Gymnasium, RL generalization, reward design, Reward Machines, temporal task specification, and synthetic data.

## What Is Now Implemented

The repo now includes several upgrades that used to live in this plan:

- held-out `train` / `eval` splits
- adaptive open-world profile families
- explicit truncation / time-budget semantics
- structured reward signals for Tasks 3-5
- tradeoff weights and objective breakdowns in observations
- Task 5 with Reward-Machine-style subgoal ordering
- a direct adaptation benchmark script for held-out Task 5 profile families

## Highest-Value Next Steps

1. **In-episode workload shifts**
   - Today workload pressure is still mostly sampled at reset time.
   - Next version should inject bursts during the episode.
   - This would make orchestration less scriptable and more adaptive.

2. **True schema-migration actions**
   - Today the benchmark mostly uses rename/cast/refresh patterns.
   - Next version should add actions like:
     - reconcile renamed field mappings
     - upgrade a downstream schema version
     - approve/reject a breaking schema change

3. **More open-world failure families**
   - unseen enum values
   - column splits/merges
   - partial source outages
   - unit drift between upstream providers
   - late-arriving corrections that invalidate a previous summary

4. **Profile-conditioned benchmark suite**
   - publish benchmark results by scenario profile, not only by task id
   - this would let us say which policies are robust to which failure families

5. **Learned policy baseline**
   - the repo now has random, heuristic, hybrid, and strict pure-LLM modes
   - the next meaningful step is a learned policy
   - even a simple imitation or offline RL baseline would make Mario look more research-grade

6. **Synthetic-data utility audit**
   - compare whether policy ranking is stable when scenario distributions change
   - if ranking stays stable, synthetic scenarios are doing their job as benchmark data

7. **Task 5 evolution beyond hand-authored reward machines**
   - Task 5 now exists and is formalized.
   - The next step is to move from hand-authored temporal progression to richer dynamics:
     - mid-episode workload shifts
     - corrections that invalidate a previous rollup
     - optional alternate valid recovery paths
     - task specifications that are still explicit, but less scripted

## What Not To Change Casually

- keep dense reward shaping
- keep the clean OpenEnv API
- keep strict pure-LLM as a credibility mode
- do not overload the benchmark with dozens of actions unless the observation space is improved alongside it

## The North Star

The papers point toward a stronger Mario that behaves less like a scripted cleanup benchmark and more like a small, adaptive ETL operations laboratory:

- changing workloads
- changing schemas
- delayed and partial data arrival
- multiple valid but cost-sensitive recovery paths
- reproducible reporting across scenario families
- compositional temporal tasks with explicit but auditable subgoal structure
