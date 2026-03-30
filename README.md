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

Mario the Plumber is an OpenEnv environment where an agent repairs broken ETL tables step by step. The environment uses a fixed discrete action space, quality-signal observations, and deterministic grading against ground truth.

## Why This Environment Matters

Real data systems break in ways that are messy but structured: missing values, duplicate records, bad types, and broken derived columns. Mario the Plumber turns that into an agent benchmark where the model has to inspect evidence, choose repair actions in the right order, and avoid making one table better while making the overall pipeline worse.

This makes it useful for evaluating agents that claim to do data engineering or operations work, because success depends on repair sequencing, not just one-shot text generation.

## What Is Implemented

- typed action, observation, and state models
- Synthetic generators for all 3 tasks
- Deterministic graders for single-table and multi-table scoring
- OpenEnv server environment with `reset`, `step`, and `state`
- Extra FastAPI endpoints: `/tasks`, `/grader`, and `/baseline`
- Typed client in [`client.py`](client.py)

## Tasks

1. Task 1: single table missing values
2. Task 2: single table duplicates and type violations
3. Task 3: multi-table cascading failures across `orders`, `customers`, and `products`

## What Makes It Hard

- actions are discrete, so the agent must pick the right repair instead of directly editing rows
- some fixes are only safe after earlier cleanup, like filling nulls before casting to integers
- Task 3 is cross-table: cleaning one table is not enough if downstream calculations still depend on broken inputs
- committing too early can lock in a worse overall score

## Action Model

```json
{
  "action_id": 3,
  "target_column": "age"
}
```

- `action_id` is required and must be `0-15`
- `target_column` is required for actions `3-9`, `11`, `12`
- `new_name` is required for action `12`
- `column_order` is required for action `13`
- Action `0` can optionally use `target_column` as a table switch in task 3

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

[`inference.py`](inference.py) follows the submission shape from the platform:

- uses the OpenAI client for LLM calls
- reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`
- runs all 3 tasks with fixed `seed=42`
- lets the model choose among safe candidate repairs on easier tasks, with heuristic fallback for invalid or premature moves
- supports seed benchmarking with `python3 inference.py --seeds 1 2 3 4 5`

Example env setup:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="deepseek-ai/DeepSeek-V3-0324"
export HF_TOKEN="your-token"
python3 inference.py
```

Verified local fallback run with no credentials, using fixed `seed=42`:

- Task 1: `0.8875` in 4 steps
- Task 2: `1.0000` in 4 steps
- Task 3: `0.9820` in 8 steps
- Average: `0.9565`

Verified local LLM-backed run with `deepseek-ai/DeepSeek-V3-0324`:

- Task 1: `0.8875` in 4 steps
- Task 2: `1.0000` in 4 steps
- Task 3: `0.9820` in 8 steps
- Average: `0.9565`
- Runtime: about `12.27s`

Example multi-seed benchmark output now shows score variance across scenarios:

- seed `1`: average `0.9690`
- seed `2`: average `0.9148`
- seed `3`: average `0.9148`

## Validation

- `openenv validate`
- [`scripts/validate-submission.sh`](scripts/validate-submission.sh)

## Evaluation Snapshot

- deterministic graders return scores in `0.0-1.0`
- success thresholds are `0.85`, `0.80`, and `0.75`
- local validation is currently passing
- the remaining high-value pre-submission check is the live HF Space validator run

## Current Local Status

- `openenv validate` passes from the repo root
- `python3 inference.py` runs all 3 official tasks with `seed=42`
- The deployed Hugging Face Space is live and responds to `/health` and `/reset`
- The deterministic fallback baseline is intended for smoke testing when model credentials are absent
- The preferred submission path is still the OpenAI-client baseline with `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`
