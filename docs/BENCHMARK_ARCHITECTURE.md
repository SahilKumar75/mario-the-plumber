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

Mario uses a benchmark-oriented internal split:

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
- `benchmark/env_actions.py`
  - repair and orchestration action handlers
- `benchmark/env_reporting.py`
  - observation packaging
  - episode reporting
  - structured progress signals
- `server/data_generator.py`
  - deterministic synthetic scenario generation for tasks 1-5
- `server/pipeline_doctor_environment.py`
  - environment lifecycle orchestration
  - state transitions
- `server/app.py`
  - OpenEnv server entrypoint
  - benchmark metadata routes
  - benchmark demo wiring

## Runtime Framing

Mario exposes three benchmark-oriented runtime framings in metadata and demo surfaces:

- `benchmark`
  - default scoring/evaluation mode
- `incident`
  - benchmark episodes framed as ETL incident-response sessions
- `hybrid`
  - benchmark mode with richer visualization/reporting

These modes do not change the public task or action contract.

## Task Progression

- Task 1: single-table missing value and format repair
- Task 2: duplicates, dtype drift, and outlier cleanup
- Task 3: cascading multi-table dependency repair
- Task 4: incremental recovery under backlog, freshness, and resource pressure
- Task 5: temporal recovery with formal subgoals and held-out profile adaptation

## Evaluation Signals

Mario keeps a scalar OpenEnv reward, while hard tasks also expose:

- `reward_breakdown`
- `objective_breakdown`
- `tradeoff_weights`
- `subgoal_progress`
- `reward_machine_state`

This keeps the environment fully OpenEnv-compatible while making evaluation easier to inspect.
