# PipelineDoctor — Complete Project Context v2

> This is the authoritative reference for building PipelineDoctor.
> Every ambiguity from v1 is resolved here with exact specs.

---

## 1. Hackathon Overview

**Event:** OpenEnv Hackathon — Meta x Scaler x Hugging Face x PyTorch
**Round 1 Deadline:** April 5, 2026 — 11:59 PM IST
**Grand Finale:** April 25-26, 2026 — Scaler School of Technology, Bangalore
**Prize Pool:** $30,000 USD | 1st Prize: $7,500 + direct Meta/HF interview

### Team
| Name | Email | Role |
|------|-------|------|
| Manu Rana | manurana26770@gmail.com | Team Lead |
| Manish Rana | manishbadm0725@gmail.com | Member |
| Sahil Kumar Singh | sahilsnowbyte@gmail.com | Member |

### What The Hackathon Asks
Build a complete, real-world OpenEnv environment that an AI agent can learn from through `step()` / `reset()` / `state()` API.

You build the GAME BOARD and RULES. The AI agent is provided by Meta's evaluators.

### Judging Criteria
| Parameter | Weight | What Judges Look For |
|-----------|--------|---------------------|
| Real-world utility | 30% | Does it model a genuine task? |
| Task & grader quality | 25% | Clear objectives? Accurate graders? Difficulty progression? |
| Environment design | 20% | Clean state management, sensible spaces, good reward shaping |
| Code quality & spec compliance | 15% | Follows OpenEnv spec, typed models, Dockerfile works |
| Creativity & novelty | 10% | Novel domain, interesting mechanics |

### Judging Phases
- **Phase 1:** Automated validation — pass/fail gate
- **Phase 2:** LLM agent runs against your environment, scored
- **Phase 3:** Human review by Meta and Hugging Face engineers

### Disqualification Criteria
- Environment does not deploy or respond
- Plagiarized or trivially modified existing environments
- Graders that always return the same score
- No baseline inference script

### What To Submit
- Public GitHub repo with all code
- `requirements.txt`
- `inference.py` — baseline agent using OpenAI client
- `README.md` — description, action/obs spaces, setup, baseline scores
- Deployed Hugging Face Spaces URL

### Framework Required
**OpenEnv framework ONLY. Never gymnasium or gym.**

---

## 2. Project: PipelineDoctor

**Name:** PipelineDoctor
**Tagline:** An RL environment where AI agents diagnose and fix broken ETL data pipelines.

### The Problem
Data pipelines break constantly. Missing values, wrong types, duplicate records, silent calculation errors. Fixing them consumes 20-40% of a data engineer's time. Current AI solves only 3.9% of pipeline tasks (ELT-Bench 2025). No RL benchmark for this domain exists anywhere.

### Realistic Score Projection
| Criteria | Weight | Score |
|----------|--------|-------|
| Real-world utility | 30% | 27/30 |
| Task & grader quality | 25% | 23/25 |
| Environment design | 20% | 18/20 |
| Code quality | 15% | 14/15 |
| Creativity | 10% | 9/10 |
| **TOTAL** | **100%** | **91/100** |

---

## 3. Exact Action Payload Schema (Gap #1 Fixed)

Actions are NOT just integers. Each action is a Pydantic model with `action_id` plus optional parameters.

### The PipelineAction Model

```python
from pydantic import BaseModel
from typing import Optional

class PipelineAction(BaseModel):
    action_id: int                            # Required — 0 to 15
    target_column: Optional[str] = None       # Required for actions 3-13
    new_name: Optional[str] = None            # Required for action 12 only
    column_order: Optional[list[str]] = None  # Required for action 13 only
```

### Which Actions Need Which Parameters

| Action ID | Action Name | target_column | new_name | column_order |
|-----------|------------|---------------|----------|--------------|
| 0 | inspect_schema | not needed | not needed | not needed |
| 1 | view_error_log | not needed | not needed | not needed |
| 2 | sample_data | not needed | not needed | not needed |
| 3 | fill_mean | required | not needed | not needed |
| 4 | fill_median | required | not needed | not needed |
| 5 | fill_forward | required | not needed | not needed |
| 6 | drop_nulls | required | not needed | not needed |
| 7 | cast_to_int | required | not needed | not needed |
| 8 | cast_to_float | required | not needed | not needed |
| 9 | cast_to_string | required | not needed | not needed |
| 10 | remove_duplicates | not needed (whole table) | not needed | not needed |
| 11 | drop_outliers | required | not needed | not needed |
| 12 | rename_column | required (old name) | required | not needed |
| 13 | reorder_columns | not needed | not needed | required |
| 14 | validate_schema | not needed | not needed | not needed |
| 15 | commit_changes | not needed | not needed | not needed |

### Validation Rule
If a required parameter is missing, `step()` returns reward=-0.1 and action_result="invalid: missing required parameter".

### Example Action Payloads

```python
PipelineAction(action_id=3, target_column="age")               # fill_mean on age column
PipelineAction(action_id=7, target_column="salary")            # cast salary to int
PipelineAction(action_id=12, target_column="custmer", new_name="customer")  # rename
PipelineAction(action_id=13, column_order=["id","name","age"]) # reorder
PipelineAction(action_id=10)                                   # remove_duplicates (no params)
PipelineAction(action_id=15)                                   # commit_changes (no params)
```

---

## 4. sample_data Action Clarification (Gap #2 Fixed)

**Decision:** `sample_data` (action_id=2) returns raw rows — but only as an action result, NOT part of the standard observation.

- Standard `PipelineObservation` NEVER contains raw rows — only quality signals.
- When agent calls action_id=2, the `action_result` field in observation is populated with 5 rows as a string.
- After all other actions, `action_result` is empty string "".

```python
class PipelineObservation(BaseModel):
    # Standard quality signals — always present
    missing_rate: float
    duplicate_rate: float
    type_violations: int
    outlier_count: int
    schema_report: dict
    recent_errors: list[str]
    current_score: float
    steps_taken: int
    stage: str
    available_actions: list[int]
    
    # Action result — populated only after certain actions
    # action_id=2 (sample_data): 5 rows as string
    # action_id=0 (inspect_schema): schema as string
    # action_id=1 (view_error_log): error list as string
    # all other actions: ""
    action_result: str = ""
```

---

## 5. Task 3 Multi-Table Scoring (Gap #3 Fixed)

Task 3 involves 3 tables: `orders`, `customers`, `products` with foreign key relationships.

### The 3 Tables

```python
orders    = {"order_id": int, "customer_id": int, "product_id": int, "quantity": int, "total_price": float}
customers = {"customer_id": int, "name": str, "email": str, "age": int}
products  = {"product_id": int, "product_name": str, "unit_price": float, "category": str}
```

### Multi-Table Scoring

```python
def score_task3(fixed_tables, ground_truth_tables):
    orders_score    = score_single_table(fixed_tables["orders"],    ground_truth_tables["orders"])
    customers_score = score_single_table(fixed_tables["customers"], ground_truth_tables["customers"])
    products_score  = score_single_table(fixed_tables["products"],  ground_truth_tables["products"])
    return round((0.50 * orders_score) + (0.30 * customers_score) + (0.20 * products_score), 4)
```

### Active Table
Agent works on ONE table at a time. `observation.stage` tells which table is active: "orders", "customers", or "products". Agent switches table by calling `inspect_schema` with `target_column` set to table name.

---

## 6. Wrong Calculation Bugs (Gap #4 Fixed)

Wrong calculations are derived column errors. Schema validation cannot detect them — values look valid but are numerically wrong.

### How They Are Injected

```python
# Ground truth: total_price = quantity * unit_price
# Broken:       total_price = quantity + unit_price  (wrong operator)
# Values look like valid floats — no type error
```

### How Agent Detects and Fixes

1. Agent calls `validate_schema` (action 14) — runs deep check, compares derived columns
2. `recent_errors` shows: `"total_price: 847 rows have calculation mismatch"`
3. Agent calls `commit_changes` (action 15) — grader auto-recalculates derived columns

### Known Derived Columns (hardcoded in grader)

- `total_price = quantity * unit_price` (orders table only)

---

## 7. Exact reset() Signature (Gap #5 Fixed)

```python
def reset(self, seed: Optional[int] = None, task_id: int = 1) -> StepResult:
    """
    Args:
        seed: Random seed. Same seed = identical broken scenario always.
        task_id: 1=Easy, 2=Medium, 3=Hard. Default=1.
    Returns:
        StepResult with initial observation, reward=0.0, done=False
    """
```

Baseline inference.py MUST use fixed seeds: `seed=42` for all 3 tasks for reproducibility.

---

## 8. Success Rule — Single Authoritative Definition (Gap #6 Fixed)

```python
# SUCCESS requires BOTH conditions:
# 1. current_score >= TASK_THRESHOLDS[task_id]
# 2. Agent calls commit_changes (action_id=15)

TASK_THRESHOLDS = {1: 0.85, 2: 0.80, 3: 0.75}
MAX_STEPS       = {1: 10,   2: 15,   3: 25}

# done=True when any of:
done = (
    (current_score >= threshold and action_id == 15) or  # SUCCESS
    (current_score < 0.10) or                            # FAILURE — data corrupted
    (steps_taken >= MAX_STEPS[task_id])                  # FAILURE — out of steps
)

# If agent calls commit_changes with score < threshold:
# reward=-0.5, done=True, success=False
```

---

## 9. Outliers Role In Tasks (Gap #7 Fixed)

Outliers are optional noise — NOT the primary challenge in any task.

| Task | Outlier Injection Rate | Primary Challenge |
|------|----------------------|-------------------|
| Task 1 | 20% of episodes | Missing values |
| Task 2 | 30% of episodes | Duplicates + type errors |
| Task 3 | 50% of episodes | Cascading multi-table errors |

Outlier definition: value more than 3 standard deviations from column mean.

---

## 10. Real OpenEnv API Spec (Gap #8 Fixed)

GitHub: https://github.com/meta-pytorch/OpenEnv

```python
# Server side — inherit from this
from openenv.core.environment import Environment

class PipelineDoctorEnvironment(Environment):
    def reset(self, **kwargs) -> Observation: ...
    def step(self, action: Action) -> Observation: ...
    
    @property
    def state(self) -> State: ...

# Client side — inherit from this
from openenv.core.env_client import EnvClient

class PipelineDoctorEnv(EnvClient):
    pass  # Inherits reset(), step(), state(), sync()

# StepResult — returned by both reset() and step()
class StepResult:
    observation: Observation
    reward: float
    done: bool
    info: dict
```

```bash
pip install openenv-core
openenv init pipeline_doctor   # scaffold
openenv validate               # check compliance
openenv push --repo-id USERNAME/pipeline-doctor  # deploy
```

---

## 11. Endpoint Response Schemas (Gap #9 Fixed)

### GET /tasks

```json
{
  "tasks": [
    {"task_id": 1, "name": "Single Table Missing Values", "difficulty": "easy", "success_threshold": 0.85, "max_steps": 10},
    {"task_id": 2, "name": "Single Table Duplicates and Types", "difficulty": "medium", "success_threshold": 0.80, "max_steps": 15},
    {"task_id": 3, "name": "Multi-Table Cascading Failure", "difficulty": "hard", "success_threshold": 0.75, "max_steps": 25}
  ],
  "action_schema": {
    "action_id": "int (0-15, required)",
    "target_column": "str (optional, required for actions 3-13)",
    "new_name": "str (optional, required for action 12 only)",
    "column_order": "list[str] (optional, required for action 13 only)"
  }
}
```

### POST /grader

Request: `{"task_id": 1, "episode_id": "abc-123"}`

Response:
```json
{
  "task_id": 1,
  "episode_id": "abc-123",
  "score": 0.87,
  "breakdown": {"completeness": 1.00, "validity": 0.90, "consistency": 0.80, "accuracy": 0.78},
  "success": true,
  "steps_taken": 7
}
```

### POST /baseline

Request: `{}`

Response:
```json
{
  "status": "complete",
  "results": [
    {"task_id": 1, "score": 0.82, "steps": 6, "success": true},
    {"task_id": 2, "score": 0.74, "steps": 12, "success": false},
    {"task_id": 3, "score": 0.61, "steps": 22, "success": false}
  ],
  "average_score": 0.72,
  "runtime_seconds": 45
}
```

---

## 12. state() Exact Fields (Gap #10 Fixed)

```python
class PipelineState(BaseModel):
    episode_id: str           # Unique UUID e.g. "abc-123-def"
    task_id: int              # 1, 2, or 3
    seed: Optional[int]       # None if random
    step_count: int           # Steps taken so far
    max_steps: int            # Maximum allowed for this task
    current_score: float      # Latest quality score 0.0-1.0
    initial_score: float      # Score at episode start
    best_score: float         # Highest score reached this episode
    done: bool                # Is episode over?
    success: Optional[bool]   # None until episode ends
    active_table: str         # "orders" | "customers" | "products" | "single"
    started_at: str           # ISO timestamp
```

---

## 13. Reward Model — NOT Required (Gap #11 Fixed)

Reward is a plain `float` inside `StepResult.reward`. No Pydantic Reward class needed.

The THREE Pydantic models required are:
1. `PipelineAction` — what agent sends in
2. `PipelineObservation` — what agent receives back
3. `PipelineState` — episode metadata from state()

---

## 14. Complete File Structure

```
pipeline_doctor/
├── models.py                   # PipelineAction, PipelineObservation, PipelineState
├── client.py                   # PipelineDoctorEnv(EnvClient)
├── openenv.yaml                # Environment metadata
├── pyproject.toml              # Python dependencies
├── README.md                   # Full documentation
├── inference.py                # Baseline OpenAI agent
└── server/
    ├── data_generator.py       # Generates broken DataFrames for all 3 tasks
    ├── grader.py               # score_single_table() and score_task3()
    ├── environment.py          # PipelineDoctorEnvironment — reset(), step(), state()
    ├── app.py                  # FastAPI routes
    ├── Dockerfile              # Container definition
    └── requirements.txt        # Server dependencies
```

---

## 15. Exact Scoring Formula

```python
def score_single_table(fixed_df, ground_truth_df, expected_types):
    total_cells  = len(fixed_df) * len(fixed_df.columns)
    completeness = 1.0 - (fixed_df.isnull().sum().sum() / total_cells)
    validity     = sum(1 for col in fixed_df.columns if str(fixed_df[col].dtype) == expected_types.get(col, "")) / len(fixed_df.columns)
    consistency  = 1.0 - (fixed_df.duplicated().sum() / len(fixed_df))
    accuracy     = (fixed_df.reset_index(drop=True) == ground_truth_df.reset_index(drop=True)).all(axis=1).mean()
    return round(0.20*completeness + 0.20*validity + 0.30*consistency + 0.30*accuracy, 4)
```

---

## 16. Exact Reward Formula

```python
def compute_reward(score_before, score_after, action_valid, done, success):
    reward  = 0.5 * (score_after - score_before)  # quality improvement
    reward -= 0.001                                 # step penalty
    if not action_valid: reward -= 0.1              # invalid action
    if done and success: reward += 1.0              # win bonus
    if done and not success: reward -= 0.5          # loss penalty
    return round(reward, 4)
```

---

## 17. openenv.yaml

```yaml
name: pipeline_doctor
version: "1.0.0"
description: "An RL environment for diagnosing and fixing broken ETL data pipelines"
tags: [openenv, data-engineering, etl, pipeline-debugging]
tasks:
  - id: 1
    name: "single-table-missing-values"
    difficulty: "easy"
  - id: 2
    name: "single-table-duplicates-types"
    difficulty: "medium"
  - id: 3
    name: "multi-table-cascading-failure"
    difficulty: "hard"
```

---

## 18. Day-by-Day Build Plan

| Day | Date | Task | Files |
|-----|------|------|-------|
| Day 1 | Mar 29 | scaffold + models + client | models.py, client.py, openenv.yaml, pyproject.toml |
| Day 2 | Mar 30 | data generator | server/data_generator.py, server/requirements.txt |
| Day 3 | Mar 31 | grader + environment core | server/grader.py, server/environment.py |
| Day 4 | Apr 1 | FastAPI server + Dockerfile + local test | server/app.py, server/Dockerfile |
| Day 5 | Apr 2 | inference.py baseline agent | inference.py |
| Day 6 | Apr 3 | deploy to HF + openenv validate | HF Space |
| Day 7 | Apr 4 | polish README + bug fixes | README.md |
| Day 8 | Apr 5 | submit before 11:59 PM IST | final submission |

---

## 19. Critical Warnings

| Risk | Severity | How To Avoid |
|------|----------|-------------|
| Using Gymnasium instead of OpenEnv | DISQUALIFIED | Import from openenv.core only |
| Grader always returns same score | DISQUALIFIED | Test with multiple seeds |
| inference.py wrong filename | DISQUALIFIED | Must be exactly `inference.py` in root |
| HF Space not deploying | DISQUALIFIED | Test on Day 6 |
| Runtime over 20 minutes | DISQUALIFIED | Time inference.py before submitting |
| Raw data in standard observation | SCORE DROP | Only quality signals in PipelineObservation |
| Missing action validation | SCORE DROP | Invalid actions return reward=-0.1, not crash |

---

## 20. Instructions For Coding Assistants

### What You Are Building
Data pipeline debugging environment using OpenEnv framework. Game board and rules — not the AI player.

### Build Order
1. `models.py` — PipelineAction, PipelineObservation, PipelineState
2. `server/data_generator.py` — broken DataFrames for all 3 tasks
3. `server/grader.py` — scoring functions
4. `server/environment.py` — reset(), step(), state()
5. `server/app.py` — FastAPI routes
6. `server/Dockerfile` — container
7. `client.py` — EnvClient
8. `inference.py` — baseline OpenAI agent
9. `README.md` — documentation

### Key Rules
- Framework: `openenv-core` only
- Models: Pydantic BaseModel
- Server: FastAPI
- reset() signature: `reset(self, seed=None, task_id=1)`
- Reward: plain float, no Pydantic model needed
- Baseline inference.py: use seed=42 for all 3 tasks
- Invalid actions: return reward=-0.1, never crash

---

## 21. Glossary

| Term | Plain English |
|------|--------------|
| OpenEnv | Framework by Meta and Hugging Face for AI training environments |
| ETL | Extract, Transform, Load — process of moving and cleaning data |
| Pipeline | Series of steps data flows through from source to destination |
| Agent | The AI that tries to fix the pipeline — NOT built by you |
| Environment | The simulated world — the game board |
| Episode | One complete run from reset() to done=True |
| Observation | What the agent sees — quality signals only (not raw data) |
| Action | PipelineAction with action_id and optional parameters |
| Reward | Plain float — no separate Pydantic model |
| Grader | score_single_table() and score_task3() |
| Ground truth | The perfect correct dataset the broken data should become |
| StepResult | Returned by step() and reset() — contains observation, reward, done, info |
| commit_changes | Action 15 — agent signals done fixing, episode ends |
| openenv validate | CLI command to check spec compliance before submitting |

---

*PipelineDoctor v2 — All gaps resolved — OpenEnv Hackathon 2026*
