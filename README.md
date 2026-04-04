---
title: Mario the Plumber
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - data-engineering
  - etl
---

# Mario the Plumber

Mario the Plumber is an **ELT/ETL pipeline incident fixer** delivered through OpenEnv. Agents diagnose broken ingestion and recovery states, repair upstream tables, restore downstream freshness, and decide when a pipeline is safe to commit.

## Benchmark Card

| Item | Value |
|---|---|
| Domain | ETL incident diagnosis, repair, and safe recovery |
| API | `reset()` / `step()` / `state` |
| Tasks | `5` |
| Actions | `20` discrete actions |
| Splits | `train`, `eval` |
| Runtime framings | `benchmark`, `incident`, `hybrid` |
| Hard tasks | Task 3, Task 4, Task 5 |
| Structured signals | reward breakdown, tradeoff weights, subgoal progress, reward-machine state |
| Live Space | [sahilksingh/mario-the-plumber](https://huggingface.co/spaces/sahilksingh/mario-the-plumber) |

## Quick Start

Run the server:

```bash
python3 -m server.app
```

Validate the environment:

```bash
openenv validate
```

Run the baseline:

```bash
python3 inference.py --policy-mode heuristic --split eval --seed 42
```

## Environment Loop

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#d9f0ff', 'primaryTextColor': '#0f172a', 'primaryBorderColor': '#2563eb', 'lineColor': '#0f766e', 'secondaryColor': '#dcfce7', 'tertiaryColor': '#fef3c7'}}}%%
flowchart LR
    A["reset(task, seed, split)"] --> B["Observation<br/>quality + dependency + workload signals"]
    B --> C["Choose discrete action"]
    C --> D["step(action)"]
    D --> E["Reward + score + subgoal progress"]
    E -->|repeat| B
    E --> F["commit_changes"]
    F --> G["success / failure / truncation"]
```

## Recovery Proof

![Benchmark overview](docs/assets/benchmark_overview.png)

Current local sweep from [scripts/benchmark_models.py](scripts/benchmark_models.py) over seeds `1 2`:

| Policy | Split | Avg Score | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | train | `0.4675` | `0.6259` | `0.5425` | `0.1875` | `0.6307` | `0.3508` |
| heuristic | train | `0.8461` | `0.7913` | `0.9050` | `0.5554` | `1.0000` | `0.9789` |
| random | eval | `0.4420` | `0.6259` | `0.5225` | `0.1881` | `0.5426` | `0.3306` |
| heuristic | eval | `0.7668` | `0.7913` | `0.8600` | `0.2082` | `0.9958` | `0.9789` |

Held-out Task 5 adaptation from [scripts/benchmark_adaptation.py](scripts/benchmark_adaptation.py):

- train mean: `0.9771`
- eval mean: `0.9771`
- held-out profile family mean: `0.9767`

## Difficulty Gap

![Difficulty gap](docs/assets/difficulty_gap.png)

The suite is designed so that realistic ETL incidents stay well above random behavior but remain solvable by structured recovery policies.

## Tasks

| Task | Difficulty | Incident Type | Tables |
|---|---|---|---|
| 1 | Easy | first-line ingestion repair | `single` |
| 2 | Medium | validation and event stabilization | `single` |
| 3 | Hard | referential repair and cascading recovery | `orders`, `customers`, `products` |
| 4 | Hard | on-call incremental recovery under backlog, freshness, and resource pressure | `orders`, `products`, `daily_summary` |
| 5 | Hard | temporal rollup recovery with schema evolution and late corrections | `source_orders`, `catalog`, `hourly_rollup` |

Each task now exposes an explicit ETL incident card with:

- what broke
- diagnosis signals
- recovery requirements
- unsafe commit conditions
- threshold rationale

## Observation and Actions

Observations expose:

- incident framing: `incident_type`, `incident_summary`, `diagnosis_signals`, `recovery_requirements`, `unsafe_commit_conditions`
- quality signals: `missing_rate`, `duplicate_rate`, `type_violations`, `outlier_count`, `format_issues`
- dependency and table signals: `table_health`, `dependency_alerts`, `commit_ready`
- orchestration signals: `backlog_rows`, `queue_backlog_age_minutes`, `freshness_lag_minutes`, `sla_severity`, `resource_level`, `required_resource_level`, `pending_batches`
- operational incident signals: `recent_failure_counters`, `drift_markers`, `dependency_health_summary`
- open-world signals: `scenario_profile`, `open_world_patterns`, `missing_expected_columns`, `column_alias_hints`
- episode semantics: `time_budget_remaining`, `truncated`, `done_reason`
- structured task signals for Tasks 3-5:
  - `reward_breakdown`
  - `objective_breakdown`
  - `tradeoff_weights`
  - `subgoal_progress`
  - `reward_machine_state`

Actions:

- `0`: inspect schema / switch table on multi-table tasks
- `3-5`: fill values
- `6`: drop null rows
- `7-9`: cast or normalize columns
- `10`: remove duplicates
- `11`: drop outliers
- `12`: rename column
- `13`: reorder columns
- `14`: validate schema
- `15`: commit changes
- `16-19`: resource scaling, batch prioritization, and pipeline-output refresh

## Space Demo

The Hugging Face Space serves the standard OpenEnv API and, when the web interface is enabled, a benchmark-specific visualization tab at `/web`:

- ETL incident overview
- incident/task explorer
- live diagnosis and recovery inspector
- benchmark results and adaptation artifacts
- architecture notes for reviewers

## Reward and Evaluation

![Objective weights](docs/assets/objective_weights.png)

Mario keeps a scalar OpenEnv reward, but the ETL recovery logic is now more explicit:

- Tasks 1-2 use the single-table mix: completeness, validity, consistency, accuracy
- Tasks 3-5 expose higher-level pipeline objective weights alongside the scalar score
- task cards distinguish **true recovery success** from **dense shaping terms**
- exploit checks explicitly guard against deletion-heavy repair, premature commit, cosmetic consistency, and fake recovery through resource overuse

- `reward_breakdown`
- `objective_breakdown`
- `tradeoff_weights`
- `subgoal_progress`
- `subgoal_order`
- `active_subgoal`
- `reward_machine_state`

These signals make the ETL incident fixer easier to audit without changing the standard OpenEnv API.

## Artifact Generation

Generate benchmark artifacts:

```bash
python3 scripts/benchmark_models.py --policies random heuristic --splits train eval --seeds 1 2 --format markdown
python3 scripts/benchmark_adaptation.py --policy-mode heuristic --seeds 1 2 3 4 5 6
python3 scripts/export_benchmark_metadata.py --seeds 1 2 3 4 5 6 --output docs/assets/benchmark_metadata.json
python3 scripts/generate_visuals.py
./scripts/validate-live-space.sh https://sahilksingh-mario-the-plumber.hf.space
```

## Baseline Modes

[inference.py](inference.py) supports:

- `heuristic`
- `hybrid`
- `pure-llm`

`pure-llm` is strict and does not silently borrow heuristic rescue.

Benchmark artifacts also report:

- action-source mix
- held-out profile-family behavior
- incident-family coverage
- generalization gaps between `train` and `eval`

## Deployment

Key submission files:

- [inference.py](inference.py)
- [openenv.yaml](openenv.yaml)
- [pyproject.toml](pyproject.toml)
- [requirements.txt](requirements.txt)
- [server/app.py](server/app.py)
- [server/Dockerfile](server/Dockerfile)

## Project Structure

- [server/pipeline_doctor_environment.py](server/pipeline_doctor_environment.py): environment lifecycle and episode orchestration
- [server/data_generator.py](server/data_generator.py): synthetic scenario generation
- [benchmark/grading.py](benchmark/grading.py): deterministic scoring and reward shaping
- [benchmark/policies/engine.py](benchmark/policies/engine.py): baseline policy orchestration
- [server/benchmark_demo.py](server/benchmark_demo.py): custom web demo
- [server/app.py](server/app.py): OpenEnv app wiring and benchmark routes

## Known Limitations

- `drop_nulls` changes row count, so the accuracy metric strongly discourages deletion-heavy repairs.
- `inference.py` is a benchmark baseline family, not a learned RL policy.
- Task 5 uses a hand-authored formal subgoal structure rather than a learned task specification.
- Mario models realistic ETL incident structure, not real schedulers, warehouses, or enterprise-scale row volume.

## Additional Docs

- [Benchmark architecture](docs/BENCHMARK_ARCHITECTURE.md)
- [Reward and adaptation](docs/REWARD_STRUCTURE_AND_ADAPTATION.md)
