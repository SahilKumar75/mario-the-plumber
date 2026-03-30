# PipelineDoctor — Complete Project Context

> Share this file with any coding assistant to get started immediately.

---

## 1. Hackathon Overview

**Event:** OpenEnv Hackathon — Meta x Scaler x Hugging Face x PyTorch
**Round 1 Deadline:** April 5, 2026 — 11:59 PM IST
**Grand Finale:** April 25-26, 2026 — Scaler School of Technology, Bangalore
**Prize Pool:** $30,000 USD
**1st Prize:** $7,500 + direct interview at Meta & Hugging Face AI teams

### Team
| Name | Email | Role |
|------|-------|------|
| Manu Rana | manurana26770@gmail.com | Team Lead |
| Manish Rana | manishbadm0725@gmail.com | Member |
| Sahil Kumar Singh | sahilsnowbyte@gmail.com | Member |

### What The Hackathon Asks
Build a complete, real-world OpenEnv environment that an AI agent can learn from through the standard `step()` / `reset()` / `state()` API.

**Plain English:** You are building the GAME BOARD and RULES. NOT the AI player. The AI agent is provided by Meta's evaluators. Your job is to build the environment the agent plays in, with clear scoring so it knows if it did well or badly.

### Judging Criteria
| Parameter | Weight | What Judges Look For |
|-----------|--------|---------------------|
| Real-world utility | 30% | Does it model a genuine task? Would someone actually use this? |
| Task & grader quality | 25% | Clear objectives? Accurate graders? Difficulty progression? |
| Environment design | 20% | Clean state management, sensible action/observation spaces, good reward shaping |
| Code quality & spec compliance | 15% | Follows OpenEnv spec, typed models, Dockerfile works |
| Creativity & novelty | 10% | Novel domain, interesting mechanics, clever reward design |

### Judging Process
- **Phase 1 — Automated Validation:** Pass/fail gate. HF Space deploys, OpenEnv spec compliance, Dockerfile builds, baseline reproduces, 3+ tasks with graders.
- **Phase 2 — Agentic Evaluation:** Standard Open LLM agent (e.g. Nemotron 3 Super) run against all environments, score variance check.
- **Phase 3 — Human Review:** Top submissions reviewed by Meta and Hugging Face engineers.

### Disqualification Criteria
- Environment does not deploy or respond
- Plagiarized or trivially modified existing environments
- Graders that always return the same score
- No baseline inference script

### Pre-Submission Checklist (ALL must pass)
| Requirement | What It Means |
|-------------|--------------|
| HF Space deploys | Ping to Space URL must return 200 and respond to reset() |
| OpenEnv spec compliance | openenv.yaml, typed models, step()/reset()/state() endpoints |
| Dockerfile builds | Automated docker build on submitted repo must succeed |
| Baseline reproduces | inference.py must complete without error and produce scores |
| 3+ tasks with graders | Scores must be in 0.0-1.0 range |
| /baseline endpoint | Trigger inference and return baseline score for all 3 tasks |
| /grader endpoint | Return grader score after episode completed |
| /tasks endpoint | Return list of tasks and action schema |
| Validator passes | Run openenv validate before submitting |
| Runtime under 20 min | inference.py must complete in less than 20 minutes |
| Machine constraints | Must run on vcpu=2, memory=8gb |

### What To Submit
- Public GitHub repository with all environment code
- requirements.txt
- inference.py — baseline agent using OpenAI client
- README.md — environment description, action/observation spaces, setup instructions, baseline scores
- Deployed Hugging Face Spaces URL

### Framework Required
**ALL environments must use the OpenEnv framework by Meta and Hugging Face. NOT Gymnasium. NOT custom frameworks. OpenEnv ONLY.**

---

## 2. Project: PipelineDoctor

### Name & Tagline
**Name:** PipelineDoctor
**Tagline:** An RL environment where AI agents diagnose and fix broken ETL data pipelines.

### The Problem Being Solved
In every tech company, data flows through a pipeline:
1. Customer does something (buys, clicks, signs up)
2. Data gets collected
3. Data gets cleaned and transformed (ETL — Extract, Transform, Load)
4. Clean data lands in dashboards and databases
5. Business makes decisions from that data

When this pipeline breaks — missing data, wrong types, duplicate records, silent wrong calculations — engineers see wrong numbers or zeros on dashboards. Finding and fixing the break consumes **20-40% of a data engineer's working time** and costs enterprises **$1M-$50M annually**.

**PipelineDoctor simulates this broken pipeline scenario.** The AI agent acts as the on-call data engineer who must diagnose and fix the pipeline.

### Why This Idea Wins
| Factor | Details |
|--------|---------|
| Real-world gap | No existing RL benchmark for data engineering exists anywhere |
| Industry size | ETL failures cost enterprises $1M-$50M annually |
| Meta relevance | Meta processes petabytes of data daily — judges live this problem |
| AI benchmark gap | Current AI solves only 3.9% of data pipeline tasks (ELT-Bench 2025) |
| Deterministic grading | Data quality metrics are mathematically precise — no ambiguity |
| Novelty | Zero existing OpenEnv environments in this domain |
| Competitor gap | Most teams will build email triage, cloud ops, customer support — oversaturated |

### Realistic Score Projection
| Criteria | Weight | Our Score |
|----------|--------|-----------|
| Real-world utility | 30% | 27/30 |
| Task & grader quality | 25% | 23/25 |
| Environment design | 20% | 18/20 |
| Code quality & compliance | 15% | 14/15 |
| Creativity & novelty | 10% | 9/10 |
| **TOTAL** | **100%** | **91/100** |

---

## 3. Technical Specification

### The 3 Core Functions
Every OpenEnv environment has exactly 3 functions. These are the ONLY functions judges test.

| Function | What It Does | Returns |
|----------|-------------|---------|
| `reset()` | Generates a fresh broken pipeline scenario | Observation of broken data state |
| `step(action)` | Agent takes an action, apply it, score the result | Observation + reward + done flag |
| `state()` | Returns current episode metadata | State object with episode info |

### The 3 Tasks
| Task | Difficulty | What Is Broken | Agent Must Do | Success Threshold |
|------|-----------|----------------|---------------|-------------------|
| Task 1 | Easy | Single table, missing values in 1 column | Detect missing values, apply correct imputation | Score >= 0.85 |
| Task 2 | Medium | Single table, duplicates + wrong data types in 2 columns | Remove duplicates AND fix type violations in correct order | Score >= 0.80 |
| Task 3 | Hard | Multi-table, cascading errors — nulls + duplicates + wrong types + silent wrong calculations | Trace root cause, fix in correct order, verify output matches ground truth | Score >= 0.75 |

### Action Space — 16 Discrete Actions
The agent can ONLY pick from this locked list. No freeform actions.

| Action ID | Action Name | What It Does | Category |
|-----------|------------|-------------|----------|
| 0 | inspect_schema | Show column names, types, expected types | Investigation |
| 1 | view_error_log | Show all detected errors in current data | Investigation |
| 2 | sample_data | Show 5 random rows of current data | Investigation |
| 3 | fill_mean | Fill missing numeric values with column average | Fix Missing Values |
| 4 | fill_median | Fill missing numeric values with column median | Fix Missing Values |
| 5 | fill_forward | Fill missing values with previous row value | Fix Missing Values |
| 6 | drop_nulls | Delete all rows containing any missing value | Fix Missing Values |
| 7 | cast_to_int | Convert target column to integer type | Fix Data Types |
| 8 | cast_to_float | Convert target column to float type | Fix Data Types |
| 9 | cast_to_string | Convert target column to string type | Fix Data Types |
| 10 | remove_duplicates | Delete all duplicate rows | Fix Duplicates |
| 11 | drop_outliers | Remove statistically extreme values (>3 std dev) | Fix Outliers |
| 12 | rename_column | Fix incorrectly named column | Fix Structure |
| 13 | reorder_columns | Reorder columns to match expected schema | Fix Structure |
| 14 | validate_schema | Check if current data matches expected schema | Validation |
| 15 | commit_changes | Submit final fixed dataset — ends episode | Finish |

### Observation Space — What The Agent Sees
Agent sees data quality SIGNALS, NOT raw data rows.

| Field | Type | Example | What It Means |
|-------|------|---------|---------------|
| missing_rate | float | 0.15 | 15% of values are missing |
| duplicate_rate | float | 0.08 | 8% of rows are duplicates |
| type_violations | int | 3 | 3 columns have wrong data type |
| outlier_count | int | 12 | 12 extreme values detected |
| schema_report | dict | {age: {expected: int, actual: str, count: 45}} | Per-column type mismatch details |
| recent_errors | list | ["Column age: 20 nulls", "Column date: wrong format"] | Last 3 errors detected |
| current_score | float | 0.43 | Current data quality score 0.0-1.0 |
| steps_taken | int | 4 | How many actions taken so far |
| stage | str | "validation" | Current pipeline stage |
| available_actions | list | [0,1,2,3,10] | Valid actions in current state |

### Scoring Formula — Exact Math
```python
# Step 1 — Compute 4 quality metrics
completeness = 1.0 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
validity     = sum(1 for col in df.columns if df[col].dtype == expected_types[col]) / len(df.columns)
consistency  = 1.0 - (df.duplicated().sum() / len(df))
accuracy     = (df == ground_truth).all(axis=1).mean()

# Step 2 — Weighted final score
final_score = (0.20 * completeness) + (0.20 * validity) + (0.30 * consistency) + (0.30 * accuracy)
# Returns float between 0.0 and 1.0
```

### Score Ranges
| Score | Meaning | Episode Outcome |
|-------|---------|----------------|
| 0.95 - 1.00 | Perfect fix | Episode SUCCESS |
| 0.80 - 0.94 | Good fix | Partial success |
| 0.50 - 0.79 | Partial fix | Needs more actions |
| 0.00 - 0.49 | Bad fix | Likely to fail |
| Below 0.10 | Data corrupted | Episode FAILURE — terminate |

### Reward Function — Per Step
```python
reward  = 0.5 * (score_after - score_before)  # quality improvement
reward -= 0.001                                 # step penalty (encourages efficiency)
reward -= 0.1   # if action invalid             # wrong action penalty
reward += 1.0   # if done and success           # win bonus
reward -= 0.5   # if done and not success       # loss penalty
```

### Data Generation
No real data needed. Pure Python generates synthetic broken data.

| Error Type | How Injected | Correct Fix |
|-----------|-------------|------------|
| Missing values | Randomly set N% of cells to None/NaN | fill_mean or fill_median or fill_forward |
| Wrong data types | Convert numeric column to string randomly | cast_to_int or cast_to_float |
| Duplicate rows | Randomly copy X rows and append | remove_duplicates |
| Outliers | Inject extreme values (999999, -99999) | drop_outliers |
| Wrong column names | Rename column with typo | rename_column |
| Wrong calculation | Apply wrong formula to derived column | Agent must detect via validate_schema |

Each `reset()` call randomizes: which errors appear, which columns affected, how many rows broken. This creates infinite unique scenarios from 3 task types.

---

## 4. Project File Structure

```
pipeline_doctor/
├── models.py                   # Pydantic models — Action, Observation, State
├── client.py                   # EnvClient for connecting to environment
├── openenv.yaml                # Environment metadata and config
├── pyproject.toml              # Python dependencies
├── README.md                   # Full environment documentation
├── inference.py                # Baseline AI agent using OpenAI client
└── server/
    ├── environment.py          # Core logic — reset(), step(), state()
    ├── data_generator.py       # Generates broken pipeline scenarios
    ├── grader.py               # Exact scoring formula
    ├── app.py                  # FastAPI server
    ├── Dockerfile              # Container definition
    └── requirements.txt        # Server Python dependencies
```

---

## 5. Day-by-Day Build Plan

| Day | Date | Task | Files |
|-----|------|------|-------|
| Day 1 | Mar 29 | Scaffold with openenv init, build models.py and client.py | models.py, client.py, openenv.yaml |
| Day 2 | Mar 30 | Build data generator and FastAPI server skeleton | data_generator.py, app.py, environment.py skeleton |
| Day 3 | Mar 31 | Build core environment logic — reset(), step(), state() | environment.py complete, grader.py |
| Day 4 | Apr 1 | Build Dockerfile, test locally with uv run server | Dockerfile, local testing |
| Day 5 | Apr 2 | Build inference.py — baseline AI agent | inference.py |
| Day 6 | Apr 3 | Deploy to Hugging Face Spaces, run openenv validate | HF Space deployment |
| Day 7 | Apr 4 | Polish README, fix bugs, re-test all 3 tasks | README.md, bug fixes |
| Day 8 | Apr 5 | Final submission before 11:59 PM IST | Submit HF Space URL |

---

## 6. OpenEnv Technical Requirements

### Install Commands
```bash
pip install openenv-core
pip install huggingface_hub
huggingface-cli login
openenv init pipeline_doctor
```

### OpenEnv Spec — Must Implement Exactly
- Typed Pydantic models for Action, Observation, Reward
- `step(action)` returns observation, reward, done, info
- `reset()` returns initial observation
- `state()` returns current State object
- `openenv.yaml` file with environment metadata
- Tested via `openenv validate`

### Required Endpoints
| Endpoint | What It Does |
|----------|-------------|
| /baseline | Triggers inference script, returns baseline score for all 3 tasks |
| /grader | Returns grader score after episode completed |
| /tasks | Returns list of tasks and action schema |

### Inference Script Rules
- File MUST be named `inference.py` in root directory
- Must use OpenAI Client for all LLM calls
- Must read from environment variables: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- Must produce reproducible scores on all 3 tasks
- Must complete in under 20 minutes
- Must run on vcpu=2, memory=8gb

### Deploy Command
```bash
cd pipeline_doctor
openenv push --repo-id YOUR_USERNAME/pipeline-doctor
```

---

## 7. Critical Warnings

| Risk | Severity | How To Avoid |
|------|----------|-------------|
| Using Gymnasium instead of OpenEnv | DISQUALIFIED | Import from openenv.core only — never from gymnasium |
| Grader always returns same score | DISQUALIFIED | Test with multiple scenarios — scores must vary |
| No inference.py or wrong filename | DISQUALIFIED | File must be named exactly inference.py in root |
| HF Space not deploying | DISQUALIFIED | Test deployment on Day 6 at latest |
| Vague reward function | SCORE DROP | Use exact formula from Section 3 |
| Raw data in observation | SCORE DROP | Only expose quality signals — not raw rows |
| Freeform action space | SCORE DROP | Use discrete 16-action list only |
| No README | SCORE DROP | Must include description, action/obs spaces, baseline scores |
| Runtime over 20 minutes | DISQUALIFIED | Time your inference.py before submitting |

---

## 8. Instructions For Coding Assistants

If you are a coding assistant helping build this project, read this carefully.

### What You Are Building
- A data pipeline debugging environment using the OpenEnv framework
- NOT a machine learning model — an environment that AI agents train in
- You build the game board and rules — not the player
- The AI agent will be tested separately by hackathon judges

### Framework Rules
- Use `openenv-core` package ONLY — never gymnasium or gym
- All models must be Pydantic BaseModel subclasses
- Server must be FastAPI
- Container must use Docker
- Must deploy to Hugging Face Spaces

### Code Standards
- All functions must have type hints
- All classes must have docstrings
- Grader must be deterministic — same input always gives same score
- reset() must accept random seed for reproducibility
- No hardcoded values — use config parameters
- Python 3.10, 3.11, or 3.12 only

### Build Order
1. `models.py` — define all Pydantic types first
2. `server/data_generator.py` — synthetic broken data creation
3. `server/grader.py` — exact scoring formula
4. `server/environment.py` — reset(), step(), state()
5. `server/app.py` — FastAPI routes
6. `server/Dockerfile` — container definition
7. `client.py` — EnvClient for connecting to server
8. `inference.py` — baseline agent using OpenAI API
9. `README.md` — documentation

### Success/Failure Conditions
- SUCCESS: quality_score >= 0.95 OR agent calls commit_changes with score >= task threshold
- FAILURE: quality_score < 0.10 OR steps_taken > max_steps
- Task 1 max steps: 10
- Task 2 max steps: 15
- Task 3 max steps: 25

---

## 9. Glossary

| Term | Plain English |
|------|--------------|
| OpenEnv | Framework by Meta and Hugging Face for creating AI training environments |
| ETL | Extract, Transform, Load — process of moving and cleaning data |
| Pipeline | Series of steps data flows through from source to destination |
| Agent | The AI that tries to fix the pipeline — NOT built by you |
| Environment | The simulated world you build — the game board |
| Episode | One complete run from reset() to done=True |
| Observation | What the agent sees at each step — data quality signals |
| Action | What the agent chooses to do — picked from 16 options |
| Reward | Score given to agent after each action — between -1.0 and +1.0 |
| Grader | Your scoring function that evaluates data quality |
| Ground truth | The perfect correct dataset that broken data should become |
| HF Space | Hugging Face Spaces — where you deploy your environment |
| Dockerfile | Instructions for building a container that runs your environment |
| Pydantic | Python library for defining typed data structures |
| FastAPI | Python library for building API servers |
| Dense reward | Score given every step — helps agent learn faster |
| Sparse reward | Score only given at end of episode — harder to learn from |
| Baseline script | inference.py — proves your environment works with a real agent |

---

*PipelineDoctor — OpenEnv Hackathon 2026 | Team Sahil Kumar Singh*
