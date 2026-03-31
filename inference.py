# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenAI-client baseline and benchmark harness for Mario the Plumber."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import re
import textwrap
import time
from typing import Any

from openai import OpenAI

try:
    from .models import PipelineDoctorAction, PipelineDoctorObservation
    from .server.data_generator import MAX_STEPS, TASK_THRESHOLDS
    from .server.pipeline_doctor_environment import PipelineDoctorEnvironment
except ImportError:
    from models import PipelineDoctorAction, PipelineDoctorObservation
    from server.data_generator import MAX_STEPS, TASK_THRESHOLDS
    from server.pipeline_doctor_environment import PipelineDoctorEnvironment

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
TEMPERATURE = 0.0
MAX_TOKENS = 220
FALLBACK_ACTION = PipelineDoctorAction(action_id=14)
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
      19=refresh_downstream_summary.
    - Prefer fixing missing values before integer casts.
    - Use remove_duplicates when duplicate_rate > 0.
    - Use cast_to_string to normalize mixed date formats, noisy text encodings, and whitespace drift.
    - In task 3, action 0 with target_column equal to a table name switches the active table.
    - In task 4, action 0 can also switch among orders, products, and daily_summary.
    - Read dependency_alerts and commit_ready before committing on task 3.
    - In task 4, read orchestration_alerts, backlog_rows, freshness_lag_minutes, and resource_level before committing.
    - Use action 15 only when commit_ready is true or the score is above the task threshold with no obvious remaining errors.
    """
).strip()


def run_baseline(
    seed: int = 42,
    *,
    split: str = "train",
    policy_mode: str = "hybrid",
    model_name: str | None = None,
) -> dict[str, object]:
    """Run the benchmark baseline over the three official tasks."""

    started = time.perf_counter()
    client = _build_client()
    selected_model = model_name or MODEL_NAME
    if policy_mode == "pure-llm" and (client is None or not selected_model):
        raise RuntimeError("pure-llm mode requires MODEL_NAME and HF_TOKEN/API_KEY")

    results: list[dict[str, object]] = []
    action_sources: Counter[str] = Counter()

    for task_id in sorted(TASK_THRESHOLDS):
        env = PipelineDoctorEnvironment()
        observation = env.reset(seed=seed, task_id=task_id, split=split)
        task_sources: Counter[str] = Counter()

        for _ in range(MAX_STEPS[task_id]):
            if env.state.done:
                break

            action, action_source = _choose_action(
                client,
                selected_model,
                policy_mode,
                task_id,
                env.state.step_count + 1,
                observation,
            )
            action_sources[action_source] += 1
            task_sources[action_source] += 1
            observation = env.step(action)

            if env.state.done:
                break

            if task_id in (3, 4) and _table_should_advance(task_id, env, observation):
                next_table = _next_table(env.state.active_table)
                if next_table:
                    observation = env.step(
                        PipelineDoctorAction(action_id=0, target_column=next_table)
                    )
                    action_sources["auto_table_switch"] += 1
                    task_sources["auto_table_switch"] += 1

        if not env.state.done:
            observation = env.step(PipelineDoctorAction(action_id=15))
            action_sources["forced_commit"] += 1
            task_sources["forced_commit"] += 1

        results.append(
            {
                "task_id": task_id,
                "score": round(observation.current_score, 4),
                "steps": env.state.step_count,
                "success": bool(env.state.success),
                "action_sources": dict(task_sources),
            }
        )

    average_score = round(
        sum(float(result["score"]) for result in results) / max(len(results), 1), 4
    )
    return {
        "status": "complete",
        "policy_mode": policy_mode,
        "scenario_split": split,
        "model_name": selected_model if client is not None else None,
        "results": results,
        "average_score": average_score,
        "action_source_totals": dict(action_sources),
        "runtime_seconds": round(time.perf_counter() - started, 2),
    }


def _build_client() -> OpenAI | None:
    if not API_KEY or not MODEL_NAME:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def _choose_action(
    client: OpenAI | None,
    model_name: str | None,
    policy_mode: str,
    task_id: int,
    step_number: int,
    observation: PipelineDoctorObservation,
) -> tuple[PipelineDoctorAction, str]:
    heuristic_action = _heuristic_action(task_id, observation)
    candidate_actions = _candidate_actions(task_id, observation, heuristic_action, policy_mode)
    strict_llm_mode = policy_mode == "pure-llm"

    if policy_mode == "heuristic":
        return heuristic_action, "heuristic"
    if client is None or not model_name:
        return heuristic_action, "heuristic_no_client"

    user_prompt = _build_user_prompt(task_id, step_number, observation, candidate_actions)
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        response_text = completion.choices[0].message.content or ""
        model_action = _parse_action(
            response_text,
            task_id,
            observation,
            fallback_action=FALLBACK_ACTION if strict_llm_mode else heuristic_action,
        )
        normalized_action = _normalize_candidate_action(model_action, candidate_actions)
        stabilized_action = _stabilize_action(
            policy_mode,
            task_id,
            observation,
            normalized_action,
            heuristic_action,
            candidate_actions,
        )
        if _same_action(stabilized_action, normalized_action):
            return stabilized_action, "llm"
        if _same_action(stabilized_action, heuristic_action):
            return stabilized_action, "heuristic_guardrail"
        return stabilized_action, "fallback"
    except Exception:
        if strict_llm_mode:
            return FALLBACK_ACTION, "llm_error"
        return heuristic_action, "heuristic_exception"


def _build_user_prompt(
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
        Backlog rows: {observation.backlog_rows}
        Freshness lag minutes: {observation.freshness_lag_minutes}
        Resource level: {observation.resource_level}
        Required resource level: {observation.required_resource_level}
        Workload pressure: {observation.workload_pressure}
        Pending batches: {observation.pending_batches}
        Downstream stale: {observation.downstream_stale}
        Orchestration alerts: {json.dumps(observation.orchestration_alerts)}
        Schema report: {json.dumps(observation.schema_report, sort_keys=True)}
        Recent errors: {json.dumps(observation.recent_errors)}
        Last action result: {observation.action_result or ""}
        Available actions: {observation.available_actions}
        Candidate next actions: {json.dumps(candidate_payloads, sort_keys=True)}

        Return one JSON object only.
        """
    ).strip()


def _parse_action(
    response_text: str,
    task_id: int,
    observation: PipelineDoctorObservation,
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


def _stabilize_action(
    policy_mode: str,
    task_id: int,
    observation: PipelineDoctorObservation,
    model_action: PipelineDoctorAction,
    heuristic_action: PipelineDoctorAction,
    candidate_actions: list[PipelineDoctorAction],
) -> PipelineDoctorAction:
    if policy_mode == "pure-llm":
        if not _action_has_required_fields(model_action):
            return FALLBACK_ACTION
        return model_action

    if not _action_has_required_fields(model_action):
        return heuristic_action

    if task_id == 3 and heuristic_action.action_id != 14:
        return heuristic_action

    if model_action.action_id == 15 and _table_needs_attention(observation):
        return heuristic_action

    if model_action.action_id == 14 and _table_needs_attention(observation):
        return heuristic_action

    if task_id == 3 and model_action.action_id == 15 and not observation.commit_ready:
        return heuristic_action

    if task_id in (1, 2) and not _is_candidate_action(model_action, candidate_actions):
        return heuristic_action

    return model_action


def _action_has_required_fields(action: PipelineDoctorAction) -> bool:
    if action.action_id in {3, 4, 5, 6, 7, 8, 9, 11, 12} and not action.target_column:
        return False
    if action.action_id == 12 and not action.new_name:
        return False
    if action.action_id == 13 and not action.column_order:
        return False
    return True


def _heuristic_action(
    task_id: int,
    observation: PipelineDoctorObservation,
) -> PipelineDoctorAction:
    if task_id == 3:
        return _task3_heuristic_action(observation)
    if task_id == 4:
        return _task4_heuristic_action(observation)

    error_text = " | ".join(observation.recent_errors).lower()
    mismatch = _first_schema_mismatch(observation)

    if observation.duplicate_rate > 0:
        return PipelineDoctorAction(action_id=10)

    null_column = _column_from_errors(error_text, "null")
    if null_column:
        if "expected int64" in error_text or "expected float64" in error_text:
            return PipelineDoctorAction(action_id=4, target_column=null_column)
        return PipelineDoctorAction(action_id=5, target_column=null_column)

    if mismatch:
        column, info = mismatch
        expected = info.get("expected")
        if expected == "int64":
            return PipelineDoctorAction(action_id=7, target_column=column)
        if expected == "float64":
            return PipelineDoctorAction(action_id=8, target_column=column)
        if expected == "object":
            return PipelineDoctorAction(action_id=9, target_column=column)

    format_column = _column_from_errors(error_text, "format mismatch")
    if format_column:
        return PipelineDoctorAction(action_id=9, target_column=format_column)

    if observation.outlier_count > 0:
        outlier_column = _column_from_errors(error_text, "outlier")
        if outlier_column:
            return PipelineDoctorAction(action_id=11, target_column=outlier_column)

    if observation.current_score >= TASK_THRESHOLDS[task_id] and not _table_needs_attention(observation):
        return PipelineDoctorAction(action_id=15)

    return FALLBACK_ACTION


def _candidate_actions(
    task_id: int,
    observation: PipelineDoctorObservation,
    heuristic_action: PipelineDoctorAction,
    policy_mode: str,
) -> list[PipelineDoctorAction]:
    if task_id == 3:
        return _task3_candidate_actions(observation, heuristic_action, policy_mode)
    if task_id == 4:
        return _task4_candidate_actions(observation, heuristic_action, policy_mode)

    error_text = " | ".join(observation.recent_errors).lower()
    mismatch = _first_schema_mismatch(observation)

    if observation.duplicate_rate > 0:
        return [PipelineDoctorAction(action_id=10)]

    null_column = _column_from_errors(error_text, "null")
    if null_column:
        expected = observation.schema_report.get(null_column, {}).get("expected")
        if expected == "int64":
            return [
                PipelineDoctorAction(action_id=4, target_column=null_column),
                PipelineDoctorAction(action_id=5, target_column=null_column),
            ]
        if expected == "float64":
            return [
                PipelineDoctorAction(action_id=3, target_column=null_column),
                PipelineDoctorAction(action_id=4, target_column=null_column),
                PipelineDoctorAction(action_id=5, target_column=null_column),
            ]
        return [PipelineDoctorAction(action_id=5, target_column=null_column)]

    if mismatch:
        column, info = mismatch
        expected = info.get("expected")
        if expected == "int64":
            return [PipelineDoctorAction(action_id=7, target_column=column)]
        if expected == "float64":
            return [PipelineDoctorAction(action_id=8, target_column=column)]
        if expected == "object":
            return [PipelineDoctorAction(action_id=9, target_column=column)]

    format_column = _column_from_errors(error_text, "format mismatch")
    if observation.format_issues > 0 and format_column:
        return [PipelineDoctorAction(action_id=9, target_column=format_column)]

    outlier_column = _column_from_errors(error_text, "outlier")
    if observation.outlier_count > 0 and outlier_column:
        return [PipelineDoctorAction(action_id=11, target_column=outlier_column)]

    if observation.current_score >= TASK_THRESHOLDS[task_id] and not _table_needs_attention(observation):
        return [PipelineDoctorAction(action_id=15), PipelineDoctorAction(action_id=14)]

    if policy_mode == "pure-llm":
        return [PipelineDoctorAction(action_id=14)]

    return [heuristic_action]


def _task3_candidate_actions(
    observation: PipelineDoctorObservation,
    heuristic_action: PipelineDoctorAction,
    policy_mode: str,
) -> list[PipelineDoctorAction]:
    if policy_mode == "hybrid":
        return [heuristic_action]

    actions: list[PipelineDoctorAction] = []
    error_text = " | ".join(observation.recent_errors).lower()
    mismatch = _first_schema_mismatch(observation)

    if policy_mode != "pure-llm" and heuristic_action.action_id != FALLBACK_ACTION.action_id:
        actions.append(heuristic_action)

    if observation.duplicate_rate > 0:
        actions.append(PipelineDoctorAction(action_id=10))

    null_column = _column_from_errors(error_text, "null")
    if null_column:
        actions.extend(
            [
                PipelineDoctorAction(action_id=4, target_column=null_column),
                PipelineDoctorAction(action_id=5, target_column=null_column),
            ]
        )

    format_column = _column_from_errors(error_text, "format mismatch")
    if observation.format_issues > 0 and format_column:
        actions.append(PipelineDoctorAction(action_id=9, target_column=format_column))

    if mismatch:
        column, info = mismatch
        expected = info.get("expected")
        if expected == "int64":
            actions.append(PipelineDoctorAction(action_id=7, target_column=column))
        elif expected == "float64":
            actions.append(PipelineDoctorAction(action_id=8, target_column=column))
        elif expected == "object":
            actions.append(PipelineDoctorAction(action_id=9, target_column=column))

    outlier_column = _column_from_errors(error_text, "outlier")
    if observation.outlier_count > 0 and outlier_column:
        actions.append(PipelineDoctorAction(action_id=11, target_column=outlier_column))

    next_table = _next_table(observation.stage)
    if next_table:
        actions.append(PipelineDoctorAction(action_id=0, target_column=next_table))

    actions.append(PipelineDoctorAction(action_id=14))
    if observation.commit_ready or observation.current_score >= TASK_THRESHOLDS[3]:
        actions.append(PipelineDoctorAction(action_id=15))

    deduped: list[PipelineDoctorAction] = []
    seen: set[str] = set()
    for action in actions:
        key = json.dumps(action.model_dump(exclude_none=True), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(action)
    return deduped or [heuristic_action]


def _task4_candidate_actions(
    observation: PipelineDoctorObservation,
    heuristic_action: PipelineDoctorAction,
    policy_mode: str,
) -> list[PipelineDoctorAction]:
    if policy_mode == "hybrid":
        return [heuristic_action]

    actions: list[PipelineDoctorAction] = []
    error_text = " | ".join(observation.recent_errors).lower()
    mismatch = _first_schema_mismatch(observation)

    if policy_mode != "pure-llm" and heuristic_action.action_id != FALLBACK_ACTION.action_id:
        actions.append(heuristic_action)

    if observation.stage == "orders":
        if observation.backlog_rows > 0:
            if observation.resource_level < observation.required_resource_level:
                actions.append(PipelineDoctorAction(action_id=16))
            actions.append(PipelineDoctorAction(action_id=18))
        null_column = _column_from_errors(error_text, "null")
        if null_column:
            actions.extend(
                [
                    PipelineDoctorAction(action_id=4, target_column=null_column),
                    PipelineDoctorAction(action_id=5, target_column=null_column),
                ]
            )
    if observation.duplicate_rate > 0:
        actions.append(PipelineDoctorAction(action_id=10))

    format_column = _column_from_errors(error_text, "format mismatch")
    if observation.format_issues > 0 and format_column:
        actions.append(PipelineDoctorAction(action_id=9, target_column=format_column))

    if mismatch:
        column, info = mismatch
        expected = info.get("expected")
        if expected == "int64":
            actions.append(PipelineDoctorAction(action_id=7, target_column=column))
        elif expected == "float64":
            actions.append(PipelineDoctorAction(action_id=8, target_column=column))
        elif expected == "object":
            actions.append(PipelineDoctorAction(action_id=9, target_column=column))

    outlier_column = _column_from_errors(error_text, "outlier")
    if observation.outlier_count > 0 and outlier_column:
        actions.append(PipelineDoctorAction(action_id=11, target_column=outlier_column))

    if observation.stage != "daily_summary":
        next_table = _next_table(observation.stage, task_id=4)
        if next_table:
            actions.append(PipelineDoctorAction(action_id=0, target_column=next_table))

    if observation.stage == "daily_summary":
        actions.append(PipelineDoctorAction(action_id=19))
    actions.append(PipelineDoctorAction(action_id=14))
    if observation.commit_ready or observation.current_score >= TASK_THRESHOLDS[4]:
        actions.append(PipelineDoctorAction(action_id=15))

    deduped: list[PipelineDoctorAction] = []
    seen: set[str] = set()
    for action in actions:
        key = json.dumps(action.model_dump(exclude_none=True), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(action)
    return deduped or [heuristic_action]


def _normalize_candidate_action(
    model_action: PipelineDoctorAction,
    candidate_actions: list[PipelineDoctorAction],
) -> PipelineDoctorAction:
    for candidate in candidate_actions:
        if candidate.action_id != model_action.action_id:
            continue

        payload = candidate.model_dump(exclude_none=True)
        update_needed = False

        if candidate.target_column and not model_action.target_column:
            payload["target_column"] = candidate.target_column
            update_needed = True
        if candidate.new_name and not model_action.new_name:
            payload["new_name"] = candidate.new_name
            update_needed = True
        if candidate.column_order and not model_action.column_order:
            payload["column_order"] = candidate.column_order
            update_needed = True

        if update_needed:
            return PipelineDoctorAction(**payload)

    return model_action


def _is_candidate_action(
    model_action: PipelineDoctorAction,
    candidate_actions: list[PipelineDoctorAction],
) -> bool:
    model_payload = model_action.model_dump(exclude_none=True)
    return any(
        candidate.model_dump(exclude_none=True) == model_payload
        for candidate in candidate_actions
    )


def _task3_heuristic_action(
    observation: PipelineDoctorObservation,
) -> PipelineDoctorAction:
    error_text = " | ".join(observation.recent_errors).lower()
    mismatch = _first_schema_mismatch(observation)

    if observation.stage == "orders":
        if observation.duplicate_rate > 0:
            return PipelineDoctorAction(action_id=10)

        format_column = _column_from_errors(error_text, "format mismatch")
        if format_column:
            return PipelineDoctorAction(action_id=9, target_column=format_column)

        if mismatch:
            column, info = mismatch
            expected = info.get("expected")
            if expected == "int64":
                return PipelineDoctorAction(action_id=7, target_column=column)
            if expected == "float64":
                return PipelineDoctorAction(action_id=8, target_column=column)
            if expected == "object":
                return PipelineDoctorAction(action_id=9, target_column=column)

        if _only_calculation_mismatch(observation):
            return PipelineDoctorAction(action_id=0, target_column="customers")

    elif observation.stage == "customers":
        null_column = _column_from_errors(error_text, "null")
        if null_column:
            return PipelineDoctorAction(action_id=4, target_column=null_column)

        format_column = _column_from_errors(error_text, "format mismatch")
        if format_column:
            return PipelineDoctorAction(action_id=9, target_column=format_column)

        if mismatch:
            column, info = mismatch
            expected = info.get("expected")
            if expected == "int64":
                return PipelineDoctorAction(action_id=7, target_column=column)
            if expected == "float64":
                return PipelineDoctorAction(action_id=8, target_column=column)
            if expected == "object":
                return PipelineDoctorAction(action_id=9, target_column=column)

        if not _table_needs_attention(observation):
            return PipelineDoctorAction(action_id=0, target_column="products")

    elif observation.stage == "products":
        if observation.duplicate_rate > 0:
            return PipelineDoctorAction(action_id=10)

        format_column = _column_from_errors(error_text, "format mismatch")
        if format_column:
            return PipelineDoctorAction(action_id=9, target_column=format_column)

        if observation.outlier_count > 0:
            return PipelineDoctorAction(action_id=11, target_column="unit_price")

        if mismatch:
            column, info = mismatch
            expected = info.get("expected")
            if expected == "int64":
                return PipelineDoctorAction(action_id=7, target_column=column)
            if expected == "float64":
                return PipelineDoctorAction(action_id=8, target_column=column)
            if expected == "object":
                return PipelineDoctorAction(action_id=9, target_column=column)

        if not _table_needs_attention(observation):
            return PipelineDoctorAction(action_id=15)

    return FALLBACK_ACTION


def _task4_heuristic_action(
    observation: PipelineDoctorObservation,
) -> PipelineDoctorAction:
    error_text = " | ".join(observation.recent_errors).lower()
    mismatch = _first_schema_mismatch(observation)

    if observation.stage == "orders":
        if observation.backlog_rows > 0:
            if observation.resource_level < observation.required_resource_level:
                return PipelineDoctorAction(action_id=16)
            return PipelineDoctorAction(action_id=18)
        if observation.duplicate_rate > 0:
            return PipelineDoctorAction(action_id=10)
        format_column = _column_from_errors(error_text, "format mismatch")
        if format_column:
            return PipelineDoctorAction(action_id=9, target_column=format_column)
        if mismatch:
            column, info = mismatch
            expected = info.get("expected")
            if expected == "int64":
                return PipelineDoctorAction(action_id=7, target_column=column)
            if expected == "float64":
                return PipelineDoctorAction(action_id=8, target_column=column)
            if expected == "object":
                return PipelineDoctorAction(action_id=9, target_column=column)
        if not _table_needs_attention(observation):
            return PipelineDoctorAction(action_id=0, target_column="products")

    elif observation.stage == "products":
        if observation.duplicate_rate > 0:
            return PipelineDoctorAction(action_id=10)
        format_column = _column_from_errors(error_text, "format mismatch")
        if format_column:
            return PipelineDoctorAction(action_id=9, target_column=format_column)
        if observation.outlier_count > 0:
            outlier_column = _column_from_errors(error_text, "outlier") or "unit_price"
            return PipelineDoctorAction(action_id=11, target_column=outlier_column)
        if mismatch:
            column, info = mismatch
            expected = info.get("expected")
            if expected == "int64":
                return PipelineDoctorAction(action_id=7, target_column=column)
            if expected == "float64":
                return PipelineDoctorAction(action_id=8, target_column=column)
            if expected == "object":
                return PipelineDoctorAction(action_id=9, target_column=column)
        if not _table_needs_attention(observation):
            return PipelineDoctorAction(action_id=0, target_column="daily_summary")

    elif observation.stage == "daily_summary":
        format_column = _column_from_errors(error_text, "format mismatch")
        if format_column:
            return PipelineDoctorAction(action_id=9, target_column=format_column)
        if mismatch:
            column, info = mismatch
            expected = info.get("expected")
            if expected == "float64":
                return PipelineDoctorAction(action_id=8, target_column=column)
            if expected == "object":
                return PipelineDoctorAction(action_id=9, target_column=column)
        if observation.downstream_stale or observation.freshness_lag_minutes > 0:
            return PipelineDoctorAction(action_id=19)
        if observation.resource_level > observation.required_resource_level:
            return PipelineDoctorAction(action_id=17)
        if observation.commit_ready:
            return PipelineDoctorAction(action_id=15)

    if observation.commit_ready:
        return PipelineDoctorAction(action_id=15)
    return FALLBACK_ACTION


def _first_schema_mismatch(
    observation: PipelineDoctorObservation,
) -> tuple[str, dict[str, Any]] | None:
    for column, info in observation.schema_report.items():
        return column, info
    return None


def _column_from_errors(error_text: str, keyword: str) -> str | None:
    for chunk in error_text.split("|"):
        chunk = chunk.strip()
        if keyword not in chunk or ":" not in chunk:
            continue
        return chunk.split(":", 1)[0].strip()
    return None


def _table_needs_attention(observation: PipelineDoctorObservation) -> bool:
    return bool(
        observation.missing_rate > 0
        or observation.duplicate_rate > 0
        or observation.type_violations > 0
        or observation.outlier_count > 0
        or observation.format_issues > 0
        or (observation.stage == "orders" and observation.backlog_rows > 0)
        or (observation.stage == "daily_summary" and observation.downstream_stale)
    )


def _has_calculation_mismatch(observation: PipelineDoctorObservation) -> bool:
    return any("calculation mismatch" in error.lower() for error in observation.recent_errors)


def _only_calculation_mismatch(observation: PipelineDoctorObservation) -> bool:
    return bool(
        observation.stage == "orders"
        and observation.missing_rate == 0
        and observation.duplicate_rate == 0
        and observation.type_violations == 0
        and observation.outlier_count == 0
        and observation.format_issues == 0
        and _has_calculation_mismatch(observation)
    )


def _table_should_advance(
    task_id: int,
    env: PipelineDoctorEnvironment,
    observation: PipelineDoctorObservation,
) -> bool:
    if task_id == 3 and _only_calculation_mismatch(observation):
        return True
    if task_id == 4:
        if env.state.active_table == "orders":
            return observation.backlog_rows == 0 and not _table_needs_attention(observation)
        if env.state.active_table == "products":
            return not _table_needs_attention(observation)
        return False
    if not env.state.done and not _table_needs_attention(observation):
        return True
    if env.state.active_table == "orders" and observation.current_score >= 0.9:
        return True
    return False


def _next_table(current_table: str, task_id: int = 3) -> str | None:
    if task_id == 4:
        order = ["orders", "products", "daily_summary"]
    else:
        order = ["orders", "customers", "products"]
    if current_table not in order:
        return None
    index = order.index(current_table)
    if index + 1 >= len(order):
        return None
    return order[index + 1]


def _same_action(left: PipelineDoctorAction, right: PipelineDoctorAction) -> bool:
    return left.model_dump(exclude_none=True) == right.model_dump(exclude_none=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Mario the Plumber baseline.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for one benchmark run.")
    parser.add_argument(
        "--split",
        choices=["train", "eval"],
        default="train",
        help="Scenario distribution split to evaluate.",
    )
    parser.add_argument(
        "--policy-mode",
        choices=["heuristic", "hybrid", "pure-llm"],
        default="hybrid",
        help="Baseline decision policy.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional model override for LLM-backed modes.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Optional list of seeds to benchmark in sequence.",
    )
    args = parser.parse_args()

    if args.seeds:
        payload = {
            "status": "benchmark",
            "runs": [
                {
                    "seed": seed,
                    **run_baseline(
                        seed=seed,
                        split=args.split,
                        policy_mode=args.policy_mode,
                        model_name=args.model_name,
                    ),
                }
                for seed in args.seeds
            ],
        }
    else:
        payload = run_baseline(
            seed=args.seed,
            split=args.split,
            policy_mode=args.policy_mode,
            model_name=args.model_name,
        )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
