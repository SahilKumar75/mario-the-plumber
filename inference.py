# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenAI-client baseline for PipelineDoctor."""

from __future__ import annotations

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
    You are controlling a discrete ETL repair environment called PipelineDoctor.
    Pick exactly one next action from the available action list.

    Rules:
    - Return JSON only.
    - Use this schema:
      {"action_id": int, "target_column": str|null, "new_name": str|null, "column_order": list[str]|null}
    - Do not include markdown fences or explanations.
    - Prefer fixing missing values before integer casts.
    - Use remove_duplicates when duplicate_rate > 0.
    - In task 3, action 0 with target_column equal to a table name switches the active table.
    - Use action 14 to validate when the table looks clean.
    - Use action 15 only when the score is at or above the task threshold or there are no obvious remaining errors.
    """
).strip()


def run_baseline(seed: int = 42) -> dict[str, object]:
    """Run the submission baseline over the three official tasks."""

    started = time.perf_counter()
    client = _build_client()
    results: list[dict[str, object]] = []

    for task_id in (1, 2, 3):
        env = PipelineDoctorEnvironment()
        observation = env.reset(seed=seed, task_id=task_id)

        for _ in range(MAX_STEPS[task_id]):
            if env.state.done:
                break

            action = _choose_action(client, task_id, env.state.step_count + 1, observation)
            observation = env.step(action)

            if env.state.done:
                break

            if task_id == 3 and _table_should_advance(env, observation):
                next_table = _next_table(env.state.active_table)
                if next_table:
                    observation = env.step(
                        PipelineDoctorAction(action_id=0, target_column=next_table)
                    )

        if not env.state.done:
            observation = env.step(PipelineDoctorAction(action_id=15))

        results.append(
            {
                "task_id": task_id,
                "score": round(observation.current_score, 4),
                "steps": env.state.step_count,
                "success": bool(env.state.success),
            }
        )

    average_score = round(
        sum(float(result["score"]) for result in results) / max(len(results), 1), 4
    )
    return {
        "status": "complete",
        "results": results,
        "average_score": average_score,
        "runtime_seconds": round(time.perf_counter() - started, 2),
    }


def _build_client() -> OpenAI | None:
    if not API_KEY or not MODEL_NAME:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def _choose_action(
    client: OpenAI | None,
    task_id: int,
    step_number: int,
    observation: PipelineDoctorObservation,
) -> PipelineDoctorAction:
    if client is None or not MODEL_NAME:
        return _heuristic_action(task_id, observation)

    user_prompt = _build_user_prompt(task_id, step_number, observation)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        response_text = completion.choices[0].message.content or ""
        return _parse_action(response_text, task_id, observation)
    except Exception:
        return _heuristic_action(task_id, observation)


def _build_user_prompt(
    task_id: int,
    step_number: int,
    observation: PipelineDoctorObservation,
) -> str:
    return textwrap.dedent(
        f"""
        Task id: {task_id}
        Task threshold: {TASK_THRESHOLDS[task_id]}
        Step: {step_number}
        Active table: {observation.stage}
        Current score: {observation.current_score}
        Missing rate: {observation.missing_rate}
        Duplicate rate: {observation.duplicate_rate}
        Type violations: {observation.type_violations}
        Outlier count: {observation.outlier_count}
        Schema report: {json.dumps(observation.schema_report, sort_keys=True)}
        Recent errors: {json.dumps(observation.recent_errors)}
        Last action result: {observation.action_result or ""}
        Available actions: {observation.available_actions}

        Return one JSON object only.
        """
    ).strip()


def _parse_action(
    response_text: str,
    task_id: int,
    observation: PipelineDoctorObservation,
) -> PipelineDoctorAction:
    match = JSON_PATTERN.search(response_text)
    if not match:
        return _heuristic_action(task_id, observation)

    try:
        payload = json.loads(match.group(0))
        return PipelineDoctorAction(**payload)
    except Exception:
        return _heuristic_action(task_id, observation)


def _heuristic_action(
    task_id: int,
    observation: PipelineDoctorObservation,
) -> PipelineDoctorAction:
    if task_id == 3:
        return _task3_heuristic_action(observation)

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

    if observation.outlier_count > 0:
        outlier_column = _column_from_errors(error_text, "outlier")
        if outlier_column:
            return PipelineDoctorAction(action_id=11, target_column=outlier_column)

    if observation.current_score >= TASK_THRESHOLDS[task_id] and not _table_needs_attention(observation):
        return PipelineDoctorAction(action_id=15)

    return FALLBACK_ACTION


def _task3_heuristic_action(
    observation: PipelineDoctorObservation,
) -> PipelineDoctorAction:
    error_text = " | ".join(observation.recent_errors).lower()
    mismatch = _first_schema_mismatch(observation)

    if observation.stage == "orders":
        if observation.duplicate_rate > 0:
            return PipelineDoctorAction(action_id=10)

        if mismatch:
            column, info = mismatch
            expected = info.get("expected")
            if expected == "int64":
                return PipelineDoctorAction(action_id=7, target_column=column)
            if expected == "float64":
                return PipelineDoctorAction(action_id=8, target_column=column)

        if _only_calculation_mismatch(observation):
            return PipelineDoctorAction(action_id=0, target_column="customers")

    elif observation.stage == "customers":
        null_column = _column_from_errors(error_text, "null")
        if null_column:
            return PipelineDoctorAction(action_id=4, target_column=null_column)

        if mismatch:
            column, info = mismatch
            expected = info.get("expected")
            if expected == "int64":
                return PipelineDoctorAction(action_id=7, target_column=column)
            if expected == "float64":
                return PipelineDoctorAction(action_id=8, target_column=column)

        if not _table_needs_attention(observation):
            return PipelineDoctorAction(action_id=0, target_column="products")

    elif observation.stage == "products":
        if observation.duplicate_rate > 0:
            return PipelineDoctorAction(action_id=10)

        if observation.outlier_count > 0:
            return PipelineDoctorAction(action_id=11, target_column="unit_price")

        if mismatch:
            column, info = mismatch
            expected = info.get("expected")
            if expected == "int64":
                return PipelineDoctorAction(action_id=7, target_column=column)
            if expected == "float64":
                return PipelineDoctorAction(action_id=8, target_column=column)

        if not _table_needs_attention(observation):
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
        or observation.recent_errors
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
        and _has_calculation_mismatch(observation)
    )


def _table_should_advance(
    env: PipelineDoctorEnvironment,
    observation: PipelineDoctorObservation,
) -> bool:
    if _only_calculation_mismatch(observation):
        return True
    if not env.state.done and not _table_needs_attention(observation):
        return True
    if env.state.active_table == "orders" and observation.current_score >= 0.9:
        return True
    return False


def _next_table(current_table: str) -> str | None:
    order = ["orders", "customers", "products"]
    if current_table not in order:
        return None
    index = order.index(current_table)
    if index + 1 >= len(order):
        return None
    return order[index + 1]


def main() -> None:
    print(json.dumps(run_baseline(seed=42), indent=2))


if __name__ == "__main__":
    main()
