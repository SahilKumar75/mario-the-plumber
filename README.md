---
title: PipelineDoctor
emoji: "🩺"
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

# PipelineDoctor

PipelineDoctor is an OpenEnv environment where an agent acts like an on-call data engineer fixing broken ETL tables. The agent works through a fixed discrete action space, receives quality-signal observations, and is graded against deterministic ground truth.

## What Is Implemented

- `PipelineDoctorAction`, `PipelineDoctorObservation`, and `PipelineDoctorState`
- Synthetic generators for all 3 tasks
- Deterministic graders for single-table and multi-table scoring
- OpenEnv server environment with `reset`, `step`, and `state`
- Extra FastAPI endpoints: `/tasks`, `/grader`, and `/baseline`
- Typed client in [`client.py`](client.py)

## Tasks

1. Task 1: single table missing values
2. Task 2: single table duplicates and type violations
3. Task 3: multi-table cascading failures across `orders`, `customers`, and `products`

## Action Model

```python
PipelineDoctorAction(
    action_id=3,
    target_column="age",
)
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

Supporting docs are under [`docs/`](docs/).

## Local Run

```bash
python3 -m server.app
```

## Baseline

[`inference.py`](inference.py) follows the submission shape from the platform:

- uses the OpenAI client for LLM calls
- reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`
- runs all 3 tasks with fixed `seed=42`
- falls back to a deterministic rule-based action picker only when credentials are missing or the model call fails locally

Example env setup:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="openai/gpt-4.1-mini"
export HF_TOKEN="your-token"
python3 inference.py
```

## Validation

- `openenv validate`
- [`scripts/validate-submission.sh`](scripts/validate-submission.sh)

## Reference Docs

- [`docs/CONTEXT 3.md`](docs/CONTEXT%203.md) is the current working spec
- [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) captures the GitHub workflow
