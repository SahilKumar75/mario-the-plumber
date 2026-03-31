# Mario Benchmark Architecture

Mario is a benchmark-first OpenEnv environment for broken ELT/ETL pipeline repair and recovery.

## Public Contract

The public submission surface remains stable:

- `reset()` / `step()` / `state`
- `/tasks`
- `/grader`
- `/baseline`
- `openenv.yaml`
- task ids `1-5`
- action ids `0-19`

## Internal Structure

Mario 2.0 now uses a clearer internal split:

- `benchmark/catalog.py`
  - benchmark version
  - task thresholds and task cards
  - profile families
  - objective weights
  - formal task specs
- `benchmark/grading.py`
  - deterministic single-table and multi-table scoring
  - reward shaping
  - consistency checks
- `benchmark/policies/`
  - prompt construction
  - heuristic and hybrid policies
  - strict pure-LLM policy helpers
  - action-candidate utilities
- `server/data_generator.py`
  - deterministic synthetic scenario generation for tasks 1-5
- `server/pipeline_doctor_environment.py`
  - environment lifecycle orchestration
  - action execution
  - observation packaging
- `server/app.py`
  - OpenEnv server entrypoint
  - benchmark metadata routes
  - benchmark demo wiring

## Runtime Framing

Mario exposes three benchmark-oriented runtime framings:

- `benchmark`
  - default scoring/evaluation mode
- `incident`
  - benchmark episodes framed as ETL incident-response sessions
- `hybrid`
  - benchmark mode with richer visualization/reporting

These modes do not change the core action/task contract. They change reporting and demo framing only.

## Task Progression

- Task 1: single-table missing value and format repair
- Task 2: duplicates, dtype drift, and outlier cleanup
- Task 3: cascading multi-table dependency repair
- Task 4: incremental recovery under backlog, freshness, and resource pressure
- Task 5: temporal recovery with formal subgoals and held-out profile adaptation

## Reward and Adaptation

Mario keeps a scalar OpenEnv reward, but hard tasks also expose:

- `reward_breakdown`
- `objective_breakdown`
- `tradeoff_weights`
- `subgoal_progress`
- `reward_machine_state`

This preserves hackathon compatibility while making the benchmark easier to inspect and analyze.

