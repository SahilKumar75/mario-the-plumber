# Mario the Plumber — Authoritative Working Spec v3

This file reconciles the earlier design notes with the OpenEnv package that is actually installed in this workspace.

## Package Reality

- Server environment base class: `openenv.core.env_server.interfaces.Environment`
- Base model classes:
  - `openenv.core.env_server.types.Action`
  - `openenv.core.env_server.types.Observation`
  - `openenv.core.env_server.types.State`
- App factory: `openenv.core.env_server.http_server.create_app`
- Client class: `openenv.core.EnvClient`
- Client return type for `reset()` and `step()`: `openenv.core.client_types.StepResult`

## API Resolution

- Server implementation:
  - `reset(...) -> observation`
  - `step(action) -> observation`
  - `state` is a property
- Client implementation:
  - `env.reset(...) -> StepResult[observation]`
  - `env.step(...) -> StepResult[observation]`
  - In the installed package here, `env.state()` is a method on the client

## Resolved Action Rules

- Action `0` (`inspect_schema`) accepts optional `target_column`
  - No `target_column`: inspect current table
  - Task 3 with `target_column in {"orders","customers","products"}`: switch active table, then inspect it
- Action `6` (`drop_nulls`) is column-scoped
  - It drops rows where `target_column` is null
- `target_column` is required for actions `3-9`, `11`, and `12`
- Action `10` is whole-table deduplication
- Action `13` uses `column_order` instead of `target_column`

## Environment Scope

- Task 1: missing-value repair on one table
- Task 2: duplicates plus type repair on one table
- Task 3: three related tables with missing values, duplicates, type drift, and derived-column calculation errors

## Success Rules

- Thresholds:
  - Task 1: `0.85`
  - Task 2: `0.80`
  - Task 3: `0.75`
- Max steps:
  - Task 1: `10`
  - Task 2: `15`
  - Task 3: `25`
- Episode ends when:
  - `commit_changes` is called
  - score drops below `0.10`
  - max step budget is exhausted

## Current Scaffold Status

- The repository root now contains the real Mario the Plumber OpenEnv scaffold
- Core files added:
  - `models.py`
  - `client.py`
  - `server/data_generator.py`
  - `server/grader.py`
  - `server/pipeline_doctor_environment.py`
  - `server/app.py`
  - `inference.py`
- The baseline now uses the OpenAI client shape expected by the platform and reads:
  - `API_BASE_URL`
  - `MODEL_NAME`
  - `HF_TOKEN`
- A local heuristic fallback still exists so the script can be smoke-tested without secrets
- The platform validator script is now stored at `scripts/validate-submission.sh`
