"""Prompt construction and response parsing for Mario baselines."""

from __future__ import annotations

import json
import re
import textwrap

try:
    from ...models import PipelineDoctorAction, PipelineDoctorObservation
except ImportError:
    from models import PipelineDoctorAction, PipelineDoctorObservation

try:
    from ..catalog import TASK_THRESHOLDS
except ImportError:
    from benchmark.catalog import TASK_THRESHOLDS

JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are controlling a discrete ETL repair environment called Mario the Plumber.
    Pick exactly one next action from the available action list.

    Rules:
    - Return JSON only.
    - Use this schema:
      {"action_id": int, "target_column": str|null, "new_name": str|null, "column_order": list[str]|null}
    - Do not include markdown fences or explanations.
    - Prefer one of the provided candidate next actions when they are available.
    - Common actions:
      4=fill_median, 5=fill_forward, 7=cast_to_int, 8=cast_to_float, 9=cast_to_string,
      10=remove_duplicates, 11=drop_outliers, 14=validate_schema, 15=commit_changes,
      16=scale_resources_up, 17=scale_resources_down, 18=prioritize_incremental_batch,
      19=refresh_pipeline_outputs.
    - Prefer fixing missing values before integer casts.
    - Use remove_duplicates when duplicate_rate > 0.
    - Use cast_to_string to normalize mixed date formats, noisy text encodings, and whitespace drift.
    - In task 3, action 0 with target_column equal to a table name switches the active table.
    - In task 3, use action 19 after upstream repair to recompute orders.total_price from product pricing.
    - In task 4, action 0 can also switch among orders, products, and daily_summary.
    - In task 5, action 0 can also switch among source_orders, catalog, and hourly_rollup.
    - Read dependency_alerts and commit_ready before committing on task 3.
    - In task 4 and task 5, read orchestration_alerts, backlog_rows, freshness_lag_minutes, and resource_level before committing.
    - Read reward_machine_state, active_subgoal, and subgoal_progress before choosing the next action on tasks 3-5.
    - Use action 15 only when commit_ready is true or the score is above the task threshold with no obvious remaining errors.
    """
).strip()


def build_user_prompt(
    task_id: int,
    step_number: int,
    observation: PipelineDoctorObservation,
    candidate_actions: list[PipelineDoctorAction],
) -> str:
    candidate_payloads = [
        action.model_dump(exclude_none=True) for action in candidate_actions
    ]
    return textwrap.dedent(
        f"""
        Task id: {task_id}
        Task threshold: {TASK_THRESHOLDS[task_id]}
        Scenario split: {observation.scenario_split}
        Scenario profile: {observation.scenario_profile}
        Open-world patterns: {json.dumps(observation.open_world_patterns)}
        Step: {step_number}
        Active table: {observation.stage}
        Current score: {observation.current_score}
        Missing rate: {observation.missing_rate}
        Duplicate rate: {observation.duplicate_rate}
        Type violations: {observation.type_violations}
        Outlier count: {observation.outlier_count}
        Format issues: {observation.format_issues}
        Commit ready: {observation.commit_ready}
        Table health: {json.dumps(observation.table_health, sort_keys=True)}
        Dependency alerts: {json.dumps(observation.dependency_alerts)}
        Schema drift count: {observation.schema_drift_count}
        Observed columns: {json.dumps(observation.observed_columns)}
        Missing expected columns: {json.dumps(observation.missing_expected_columns)}
        Column alias hints: {json.dumps(observation.column_alias_hints, sort_keys=True)}
        Backlog rows: {observation.backlog_rows}
        Freshness lag minutes: {observation.freshness_lag_minutes}
        Resource level: {observation.resource_level}
        Required resource level: {observation.required_resource_level}
        Workload pressure: {observation.workload_pressure}
        Pending batches: {observation.pending_batches}
        Downstream stale: {observation.downstream_stale}
        Orchestration alerts: {json.dumps(observation.orchestration_alerts)}
        Time budget remaining: {observation.time_budget_remaining}
        Time budget ratio: {observation.time_budget_ratio}
        Truncated: {observation.truncated}
        Done reason: {observation.done_reason}
        Synthetic data notes: {json.dumps(observation.synthetic_data_notes)}
        Reward breakdown: {json.dumps(observation.reward_breakdown, sort_keys=True)}
        Objective breakdown: {json.dumps(observation.objective_breakdown, sort_keys=True)}
        Tradeoff weights: {json.dumps(observation.tradeoff_weights, sort_keys=True)}
        Subgoal progress: {json.dumps(observation.subgoal_progress, sort_keys=True)}
        Subgoal order: {json.dumps(observation.subgoal_order)}
        Active subgoal: {observation.active_subgoal}
        Reward machine state: {observation.reward_machine_state}
        Adaptation target: {observation.adaptation_target}
        Heldout profile family: {observation.heldout_profile_family}
        Schema report: {json.dumps(observation.schema_report, sort_keys=True)}
        Recent errors: {json.dumps(observation.recent_errors)}
        Last action result: {observation.action_result or ""}
        Available actions: {observation.available_actions}
        Candidate next actions: {json.dumps(candidate_payloads, sort_keys=True)}

        Return one JSON object only.
        """
    ).strip()


def parse_action(
    response_text: str,
    fallback_action: PipelineDoctorAction,
) -> PipelineDoctorAction:
    match = JSON_PATTERN.search(response_text)
    if not match:
        return fallback_action

    try:
        payload = json.loads(match.group(0))
        return PipelineDoctorAction(**payload)
    except Exception:
        return fallback_action
