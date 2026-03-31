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

Mario the Plumber is an OpenEnv benchmark for **ETL repair and online pipeline recovery**. It starts with broken tables, but the harder end of the benchmark now includes workload pressure, pending incremental batches, schema drift, stale downstream aggregates, and orchestration choices around resource scaling and refresh timing.

## Benchmark at a Glance

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#d9f0ff', 'primaryTextColor': '#0f172a', 'primaryBorderColor': '#2563eb', 'lineColor': '#0f766e', 'secondaryColor': '#dcfce7', 'tertiaryColor': '#fef3c7'}}}%%
flowchart LR
    A["reset(task, seed)"] --> B["Observation signals<br/>missing rate, duplicates, schema drift, score"]
    B --> C["Agent chooses discrete repair action"]
    C --> D["step(action)"]
    D --> E["Updated table state"]
    E --> F["Reward + current score"]
    F -->|repeat| B
    F --> G["commit_changes"]
    G --> H["Final success / failure"]
```

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ede9fe', 'primaryTextColor': '#111827', 'primaryBorderColor': '#7c3aed', 'lineColor': '#1d4ed8', 'secondaryColor': '#dbeafe', 'tertiaryColor': '#dcfce7'}}}%%
flowchart TD
    T1["Task 1<br/>single-table missing values"] --> T2["Task 2<br/>duplicates + type drift"]
    T2 --> T3["Task 3<br/>multi-table dependency repair"]
    T3 --> O["orders"]
    T3 --> C["customers"]
    T3 --> P["products"]
    P --> X["derived totals stay wrong if upstream prices stay broken"]
    O --> X
```

## Benchmark Card

| Item | Value |
|---|---|
| Domain | ETL / data quality repair + online recovery |
| API | `reset()` / `step()` / `state` |
| Tasks | 4 |
| Action space | 20 discrete actions |
| Scenario splits | `train`, `eval` |
| Policy modes | `random`, `heuristic`, `hybrid`, `pure-llm` |
| Success thresholds | `0.85`, `0.80`, `0.75`, `0.78` |
| Initial Task 3 score over 20 seeds | avg `0.2005` |
| Random Task 3 score over 20 seeds | avg `0.2065` |
| Structured Task 3 baseline | `0.9070` |
| Initial Task 4 score over 10 seeds | train avg `0.3196`, eval avg `0.3169` |
| Live Space | [`sahilksingh/mario-the-plumber`](https://huggingface.co/spaces/sahilksingh/mario-the-plumber) |

## What Changed In The New Benchmark Version

| Area | Earlier Benchmark | Current Benchmark |
|---|---|---|
| Generalization story | single fixed-seed demo | explicit `train` / `eval` splits |
| Baselines | mostly one hybrid path | `random`, `heuristic`, `hybrid`, `pure-llm` modes |
| Task 3 difficulty | random agent stayed too high | random stays near initial broken score |
| Observation design | flat error summary | table-health, dependency alerts, format issues, commit readiness, workload signals |
| Reporting | one-off runs | reproducible benchmark table via `scripts/benchmark_models.py` |

## Visuals

![Benchmark landscape](docs/assets/benchmark_landscape.png)

![Task 4 recovery curve](docs/assets/task4_recovery_curve.png)

## Why This Benchmark Matters

Real data systems fail in structured ways: missing values, schema drift, duplicate records, and broken derived fields. Mario the Plumber turns that into an agent benchmark where the model has to diagnose the failure, choose the right repair, and avoid damaging the table while fixing it.

This is useful because it tests a kind of work that production agents actually need to do:

- detect the source of a data quality regression
- choose repairs in the correct order
- reason over schema constraints instead of free-form text alone
- handle cross-table dependencies before committing a final fix
- recover a live ETL system while backlog, freshness, and resource pressure are still changing

## Why It Is Hard

The task suite is deliberately staged so the agent cannot win by emitting generic cleanup actions:

- Task 1 requires basic missing-value repair without hurting schema validity
- Task 2 mixes duplicates with type drift, so the agent has to remove redundancy and restore the expected dtypes
- Task 3 introduces cross-table reasoning, where a premature commit can recompute bad derived values from still-broken upstream data
- Task 4 adds incremental recovery, where the agent must scale resources, ingest delayed batches, normalize schema drift, refresh downstream aggregates, and only then commit

The environment also gives partial progress signals, which means the agent has to improve score steadily instead of relying on a binary pass/fail end state.

## What Is Implemented

- typed action, observation, and state models
- Synthetic generators for all 4 tasks
- train/eval scenario split for held-out evaluation
- Deterministic graders for single-table and multi-table scoring
- Task 4 orchestration features for backlog, freshness lag, and resource pressure
- OpenEnv server environment with `reset`, `step`, and `state`
- Extra FastAPI endpoints: `/tasks`, `/grader`, and `/baseline`
- Typed client in [`client.py`](client.py)

## Task Suite

1. Task 1: single table missing values
2. Task 2: single table duplicates and type violations
3. Task 3: multi-table cascading failures across `orders`, `customers`, and `products`
4. Task 4: incremental ETL recovery across `orders`, `products`, and `daily_summary`

## What Makes It Hard

- actions are discrete, so the agent must pick the right repair or orchestration move instead of directly editing rows
- some fixes are only safe after earlier cleanup, like filling nulls before casting to integers
- Task 3 is cross-table: cleaning one table is not enough if downstream calculations still depend on broken inputs
- Task 4 is operational: over time the agent must reason about backlog, freshness, resource level, and stale downstream state
- committing too early can lock in a worse overall score

## Action Model

```json
{
  "action_id": 3,
  "target_column": "age"
}
```

- `action_id` is required and must be `0-19`
- `target_column` is required for actions `3-9`, `11`, `12`
- `new_name` is required for action `12`
- `column_order` is required for action `13`
- Action `0` can optionally use `target_column` as a table switch in task 3 and task 4
- Task 4 adds orchestration actions:
  - `16`: `scale_resources_up`
  - `17`: `scale_resources_down`
  - `18`: `prioritize_incremental_batch`
  - `19`: `refresh_downstream_summary`

## Required Submission Files

This repo now uses the environment itself as the repository root. Key submission files are:

- [`inference.py`](inference.py)
- [`requirements.txt`](requirements.txt)
- [`openenv.yaml`](openenv.yaml)
- [`pyproject.toml`](pyproject.toml)
- [`uv.lock`](uv.lock)
- [`server/app.py`](server/app.py)
- [`server/Dockerfile`](server/Dockerfile)

## Local Run

```bash
python3 -m server.app
```

## Baseline

[`inference.py`](inference.py) now supports multiple policy modes:

- uses the OpenAI client for LLM calls
- reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`
- supports `heuristic`, `hybrid`, and `pure-llm` policy modes
- supports `train` and `eval` scenario splits
- records where actions came from (`llm`, `heuristic_guardrail`, `heuristic`, `auto_table_switch`)
- supports seed benchmarking with `python3 inference.py --seeds 1 2 3 4 5`

Example env setup:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="deepseek-ai/DeepSeek-V3-0324"
export HF_TOKEN="your-token"
python3 inference.py --policy-mode pure-llm --split eval
```

Example benchmark commands:

```bash
python3 inference.py --policy-mode heuristic --split train --seed 42
python3 inference.py --policy-mode heuristic --split eval --seed 42
python3 scripts/benchmark_models.py --policies random heuristic --splits train eval --seeds 1 2 3 --format markdown
```

Current local heuristic runs with `seed=42`:

- train split:
  - Task 1: `0.9250` in 4 steps
  - Task 2: `1.0000` in 4 steps
  - Task 3: `0.9820` in 12 steps
  - Task 4: `0.8000` in 11 steps
  - Average: `0.9268`
- eval split:
  - Task 1: `0.9250` in 5 steps
  - Task 2: `1.0000` in 5 steps
  - Task 3: `0.9820` in 13 steps
  - Task 4: `0.8000` in 11 steps
  - Average: `0.9091`

## Benchmark Results

| Policy | Split | Avg Score | Task 1 | Task 2 | Task 3 | Task 4 |
|---|---:|---:|---:|---:|---:|---:|
| random | train | `0.4314` | `0.6883` | `0.5433` | `0.1952` | `0.2988` |
| heuristic | train | `0.9028` | `0.9125` | `0.9667` | `0.9320` | `0.8000` |
| random | eval | `0.4278` | `0.6883` | `0.5433` | `0.1952` | `0.2844` |
| heuristic | eval | `0.9091` | `0.9125` | `0.9667` | `0.9570` | `0.8000` |

Strict `pure-llm` mode is implemented in [`inference.py`](inference.py). It now disables heuristic rescue so model-only evaluation is honest, but it should be re-benchmarked after the Task 4 upgrade with your preferred live model credentials.

## Evaluation Summary

The grading logic is deterministic and score-based rather than binary-only:

- observations expose repair signals such as missing-rate, duplicate-rate, type violations, outlier count, format mismatches, dependency alerts, per-table health summaries, backlog rows, freshness lag, workload pressure, and resource requirements
- each task has a fixed success threshold
- the reward function provides partial progress and penalizes invalid or destructive actions
- Task 3 uses weighted multi-table scoring so the agent must repair the full pipeline, not just one table

Current local thresholds:

- Task 1: `0.85`
- Task 2: `0.80`
- Task 3: `0.75`
- Task 4: `0.78`

Task 3 hardening checks now show a meaningful difficulty gap:

- initial Task 3 score over 20 seeds: min `0.2001`, max `0.2037`, avg `0.2005`
- random agent on Task 3 over 20 seeds: min `0.2001`, max `0.2112`, avg `0.2065`
- structured baseline on Task 3, seed `42`: `0.9070`

Task 4 checks show the online recovery setting is meaningfully harder than static repair:

- initial Task 4 score over 10 seeds: train avg `0.3196`, eval avg `0.3169`
- random Task 4 benchmark score: train `0.2988`, eval `0.2844`
- structured Task 4 baseline, seed `42`: `0.8000`

## Validation

- `openenv validate`
- [`scripts/validate-submission.sh`](scripts/validate-submission.sh)
- Research-grounded benchmark review: [`docs/RL_BENCHMARK_REVIEW.md`](docs/RL_BENCHMARK_REVIEW.md)

## Evaluation Snapshot

- deterministic graders return scores in `0.0-1.0`
- success thresholds are `0.85`, `0.80`, `0.75`, and `0.78`
- local validation is currently passing
- the remaining high-value pre-submission check is the live HF Space validator run

## Current Local Status

- `openenv validate` passes from the repo root
- `python3 inference.py` now runs all 4 benchmark tasks with explicit split + policy controls
- `python3 scripts/benchmark_models.py` produces reproducible benchmark tables
- The deployed Hugging Face Space is live and responds to `/health` and `/reset`
- The preferred submission path is still the OpenAI-client baseline with `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`
- The repo now supports held-out evaluation, orchestration-heavy Task 4 recovery, and strict pure-LLM benchmarking without changing the environment API

## Known Limitations

- `drop_nulls` changes row count, so the accuracy metric strongly discourages deletion-heavy repair paths; the intended agent behavior is to prefer fill and type-repair actions over row removal.
- The provided `inference.py` is a family of baselines, not a learned RL policy. `pure-llm` mode is now strict and does not borrow heuristic rescue, so its lower score should be read as a cleaner model-only benchmark rather than a submission-optimized baseline.
- Task 4 currently models one style of online ETL recovery. Future extensions should vary workload bursts during the episode rather than only at reset time.

## Additional Docs

- [Research-grounded benchmark review](docs/RL_BENCHMARK_REVIEW.md)
- [Adaptive ETL upgrade notes](docs/ADAPTIVE_ETL_UPGRADE.md)
