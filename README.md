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

Mario the Plumber is an OpenEnv benchmark for **broken ELT/ETL pipeline repair and recovery**. Agents diagnose data-quality failures, repair cross-table dependencies, clear delayed batches, restore downstream summaries, and decide when it is safe to commit a pipeline.

## Benchmark Card

| Item | Value |
|---|---|
| Domain | ETL repair + online pipeline recovery |
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

## Benchmark Results

![Benchmark overview](docs/assets/benchmark_overview.png)

Current local sweep from [scripts/benchmark_models.py](scripts/benchmark_models.py) over seeds `1 2`:

| Policy | Split | Avg Score | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | train | `0.4239` | `0.6512` | `0.5425` | `0.1900` | `0.3755` | `0.3603` |
| heuristic | train | `0.9169` | `0.9250` | `0.9750` | `0.9055` | `0.8000` | `0.9789` |
| random | eval | `0.4053` | `0.6659` | `0.5425` | `0.1931` | `0.2853` | `0.3400` |
| heuristic | eval | `0.9089` | `0.9062` | `0.9750` | `0.8920` | `0.7925` | `0.9789` |

Held-out Task 5 adaptation from [scripts/benchmark_adaptation.py](scripts/benchmark_adaptation.py):

- train mean: `0.9774`
- eval mean: `0.9774`
- held-out profile family mean: `0.9767`

## Difficulty Gap

![Difficulty gap](docs/assets/difficulty_gap.png)

The benchmark is designed so that hard tasks stay clearly above random behavior but remain solvable by structured policies.

## Tasks

| Task | Difficulty | Focus | Tables |
|---|---|---|---|
| 1 | Easy | missing values + format cleanup | `single` |
| 2 | Medium | duplicates + dtype repair | `single` |
| 3 | Hard | cross-table dependency repair | `orders`, `customers`, `products` |
| 4 | Hard | online ETL recovery under backlog, freshness, and resource pressure | `orders`, `products`, `daily_summary` |
| 5 | Hard | temporal recovery with formal subgoal structure | `source_orders`, `catalog`, `hourly_rollup` |

## Observation and Actions

Observations expose:

- quality signals: `missing_rate`, `duplicate_rate`, `type_violations`, `outlier_count`, `format_issues`
- dependency and table signals: `table_health`, `dependency_alerts`, `commit_ready`
- orchestration signals: `backlog_rows`, `freshness_lag_minutes`, `resource_level`, `required_resource_level`, `pending_batches`
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
- `16-19`: resource scaling, batch prioritization, and downstream refresh

## Space Demo

The Hugging Face Space serves the standard OpenEnv API and, when the web interface is enabled, a benchmark-specific visualization tab at `/web`:

- benchmark overview
- task explorer
- live episode inspector
- benchmark results and adaptation artifacts
- architecture notes for reviewers

## Reward and Evaluation

![Objective weights](docs/assets/objective_weights.png)

Mario returns a scalar OpenEnv reward, but the benchmark now exposes its scoring structure more clearly:

- Tasks 1-2 use the single-table mix: completeness, validity, consistency, accuracy
- Tasks 3-5 expose higher-level pipeline objective weights alongside the scalar score

- `reward_breakdown`
- `objective_breakdown`
- `tradeoff_weights`
- `subgoal_progress`
- `subgoal_order`
- `active_subgoal`
- `reward_machine_state`

These signals make the benchmark easier to audit without changing the standard OpenEnv API.

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

## Additional Docs

- [Benchmark architecture](docs/BENCHMARK_ARCHITECTURE.md)
- [Reward and adaptation](docs/REWARD_STRUCTURE_AND_ADAPTATION.md)
