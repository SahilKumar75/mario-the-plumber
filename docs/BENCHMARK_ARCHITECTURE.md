# Mario ETL Incident Architecture

Mario is an ETL/ELT pipeline incident fixer delivered through OpenEnv. The benchmark layer exists to make incident diagnosis, repair, and recovery reproducible and judgeable.

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
- `server/incidents/`
  - self-contained trace-grounded incident fixture packs
  - incident manifests, DAG-run traces, and warehouse-event traces
- `benchmark/grading.py`
  - deterministic single-table and multi-table scoring
  - reward shaping
  - consistency checks
- `benchmark/evaluation.py`
  - score dispatch
  - episode summary snapshots
- `benchmark/progress.py`
  - subgoal progress
  - reward-machine state
  - dependency-health summaries
- `benchmark/task_runtime/`
  - task-conditioned runtime progression
  - dependency-health dispatch
  - runtime error dispatch
- `benchmark/actions/`
  - table repair handlers
  - orchestration handlers
  - commit gating
- `benchmark/policies/`
  - prompt construction
  - heuristic and hybrid policies
  - strict pure-LLM policy helpers
  - action-candidate utilities
- `benchmark/env_reporting.py`
  - observation packaging
  - episode reporting
  - structured progress signals
- `server/data_generator.py`
  - scenario dispatch for tasks 1-5
- `server/pipeline_doctor_environment.py`
  - OpenEnv lifecycle shell
  - state transitions and API wiring
- `server/runtime.py`
  - episode initialization
  - step resolution
- `server/app.py`
  - OpenEnv server entrypoint
  - benchmark metadata routes
  - benchmark demo wiring

## Runtime Framing

Mario exposes three runtime framings in metadata and demo surfaces:

- `benchmark`
  - default scoring/evaluation mode
- `incident`
  - benchmark episodes framed as ETL incident-response sessions
- `hybrid`
  - benchmark mode with richer visualization/reporting

These modes do not change the public task or action contract.

## Task Progression

- Task 1: first-line ingestion repair after null and contract drift
- Task 2: validation and event stabilization after retries and dtype regressions
- Task 3: referential repair and cascading downstream recovery
- Task 4: on-call incremental recovery under backlog, freshness, and resource pressure
  - incremental backlog is replayed one batch at a time
- Task 5: temporal rollup recovery with schema evolution, late corrections, and held-out profile adaptation
  - held-out families cover schema extension drift, rollup contract drift, and correction replay drift
  - temporal closure requires replaying the affected late-correction windows before commit

## Evaluation Signals

Mario keeps a scalar OpenEnv reward, while observations also expose incident and recovery structure:

- `incident_type`
- `incident_summary`
- `diagnosis_signals`
- `recovery_requirements`
- `unsafe_commit_conditions`
- `queue_backlog_age_minutes`
- `sla_severity`
- `recent_failure_counters`
- `drift_markers`
- `dependency_health_summary`

Hard tasks also expose:

- `reward_breakdown`
- `objective_breakdown`
- `tradeoff_weights`
- `subgoal_progress`
- `reward_machine_state`

This keeps the environment fully OpenEnv-compatible while making evaluation easier to inspect.

## ETL-Native Control Surface

The public action ids still remain `0-19`, but the orchestration end of the action space now reads more like an ETL incident runbook than generic data cleaning:

- `16`: `scale_recovery_workers_up`
- `17`: `scale_recovery_workers_down`
- `18`: `replay_priority_batch`
- `19`: `refresh_downstream_assets`

On Tasks 4 and 5, `replay_priority_batch` no longer clears the entire backlog in one move. It replays only the earliest pending batch window, updates freshness and backlog metadata, and leaves remaining replay pressure visible to the agent.
