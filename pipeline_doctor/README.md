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
- Typed client in [`client.py`](/Users/sahilkumarsingh/Desktop/MARIO-the plumber/pipeline_doctor/client.py)

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

## API Notes

- On the server, the environment follows the installed OpenEnv pattern:
  - `reset(...) -> Observation`
  - `step(action) -> Observation`
  - `state` is a property
- On the client, `EnvClient.reset()` and `EnvClient.step()` return `StepResult`
- On the installed client package in this workspace, state is accessed as `env.state()`

## Local Run

```bash
cd /Users/sahilkumarsingh/Desktop/MARIO-the plumber/pipeline_doctor
python3 -m server.app
```

## Baseline

[`inference.py`](/Users/sahilkumarsingh/Desktop/MARIO-the plumber/pipeline_doctor/inference.py) now follows the submission shape from the platform:

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

The pre-submission validator script from the platform is included at [validate-submission.sh](/Users/sahilkumarsingh/Desktop/MARIO-the plumber/pipeline_doctor/scripts/validate-submission.sh).
