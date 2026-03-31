"""Heuristic, hybrid, and strict pure-LLM policy helpers."""

from __future__ import annotations

import json

from openai import OpenAI

try:
    from ...models import PipelineDoctorAction, PipelineDoctorObservation
except ImportError:
    from models import PipelineDoctorAction, PipelineDoctorObservation

try:
    from ..catalog import TASK_THRESHOLDS
    from .prompts import SYSTEM_PROMPT, build_user_prompt, parse_action
    from .utils import (
        action_has_required_fields,
        alias_fix_action,
        column_from_errors,
        first_schema_mismatch,
        next_table,
        repair_action_for_mismatch,
        same_action,
        table_needs_attention,
    )
except ImportError:
    from benchmark.catalog import TASK_THRESHOLDS
    from benchmark.policies.prompts import SYSTEM_PROMPT, build_user_prompt, parse_action
    from benchmark.policies.utils import (
        action_has_required_fields,
        alias_fix_action,
        column_from_errors,
        first_schema_mismatch,
        next_table,
        repair_action_for_mismatch,
        same_action,
        table_needs_attention,
    )

TEMPERATURE = 0.0
MAX_TOKENS = 220
FALLBACK_ACTION = PipelineDoctorAction(action_id=14)


def choose_action(
    client: OpenAI | None,
    model_name: str | None,
    policy_mode: str,
    task_id: int,
    step_number: int,
    observation: PipelineDoctorObservation,
) -> tuple[PipelineDoctorAction, str]:
    heuristic_action = heuristic_action_for(task_id, observation)
    candidate_actions = candidate_actions_for(task_id, observation, heuristic_action, policy_mode)
    strict_llm_mode = policy_mode == "pure-llm"

    if policy_mode == "heuristic":
        return heuristic_action, "heuristic"
    if client is None or not model_name:
        return heuristic_action, "heuristic_no_client"

    user_prompt = build_user_prompt(task_id, step_number, observation, candidate_actions)
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
        model_action = parse_action(
            response_text,
            FALLBACK_ACTION if strict_llm_mode else heuristic_action,
        )
        normalized_action = normalize_candidate_action(model_action, candidate_actions)
        stabilized_action = stabilize_action(
            policy_mode,
            task_id,
            observation,
            normalized_action,
            heuristic_action,
            candidate_actions,
        )
        if same_action(stabilized_action, normalized_action):
            return stabilized_action, "llm"
        if same_action(stabilized_action, heuristic_action):
            return stabilized_action, "heuristic_guardrail"
        return stabilized_action, "fallback"
    except Exception:
        if strict_llm_mode:
            return FALLBACK_ACTION, "llm_error"
        return heuristic_action, "heuristic_exception"


def stabilize_action(
    policy_mode: str,
    task_id: int,
    observation: PipelineDoctorObservation,
    model_action: PipelineDoctorAction,
    heuristic_action: PipelineDoctorAction,
    candidate_actions: list[PipelineDoctorAction],
) -> PipelineDoctorAction:
    if policy_mode == "pure-llm":
        if not action_has_required_fields(model_action):
            return FALLBACK_ACTION
        return model_action

    if not action_has_required_fields(model_action):
        return heuristic_action

    if task_id == 3 and heuristic_action.action_id != 14:
        return heuristic_action

    if model_action.action_id == 15 and table_needs_attention(observation):
        return heuristic_action

    if model_action.action_id == 14 and table_needs_attention(observation):
        return heuristic_action

    if task_id == 3 and model_action.action_id == 15 and not observation.commit_ready:
        return heuristic_action

    if task_id in (1, 2) and not is_candidate_action(model_action, candidate_actions):
        return heuristic_action

    return model_action


def heuristic_action_for(
    task_id: int,
    observation: PipelineDoctorObservation,
) -> PipelineDoctorAction:
    if task_id == 3:
        return task3_heuristic_action(observation)
    if task_id == 4:
        return task4_heuristic_action(observation)
    if task_id == 5:
        return task5_heuristic_action(observation)

    error_text = " | ".join(observation.recent_errors).lower()
    mismatch = first_schema_mismatch(observation)
    alias_action = alias_fix_action(observation)

    if alias_action:
        return alias_action
    if observation.duplicate_rate > 0:
        return PipelineDoctorAction(action_id=10)

    null_column = column_from_errors(error_text, "null")
    if null_column:
        if "expected int64" in error_text or "expected float64" in error_text:
            return PipelineDoctorAction(action_id=4, target_column=null_column)
        return PipelineDoctorAction(action_id=5, target_column=null_column)

    if mismatch:
        column, info = mismatch
        return repair_action_for_mismatch(column, info)

    format_column = column_from_errors(error_text, "format mismatch")
    if format_column:
        return PipelineDoctorAction(action_id=9, target_column=format_column)

    if observation.outlier_count > 0:
        outlier_column = column_from_errors(error_text, "outlier")
        if outlier_column:
            return PipelineDoctorAction(action_id=11, target_column=outlier_column)

    if observation.current_score >= TASK_THRESHOLDS[task_id] and not table_needs_attention(observation):
        return PipelineDoctorAction(action_id=15)

    return FALLBACK_ACTION


def candidate_actions_for(
    task_id: int,
    observation: PipelineDoctorObservation,
    heuristic_action: PipelineDoctorAction,
    policy_mode: str,
) -> list[PipelineDoctorAction]:
    if task_id == 3:
        return task3_candidate_actions(observation, heuristic_action, policy_mode)
    if task_id == 4:
        return task4_candidate_actions(observation, heuristic_action, policy_mode)
    if task_id == 5:
        return task5_candidate_actions(observation, heuristic_action, policy_mode)

    error_text = " | ".join(observation.recent_errors).lower()
    mismatch = first_schema_mismatch(observation)
    alias_action = alias_fix_action(observation)

    if alias_action:
        return [alias_action]
    if observation.duplicate_rate > 0:
        return [PipelineDoctorAction(action_id=10)]

    null_column = column_from_errors(error_text, "null")
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
        return [repair_action_for_mismatch(column, info)]

    format_column = column_from_errors(error_text, "format mismatch")
    if observation.format_issues > 0 and format_column:
        return [PipelineDoctorAction(action_id=9, target_column=format_column)]

    outlier_column = column_from_errors(error_text, "outlier")
    if observation.outlier_count > 0 and outlier_column:
        return [PipelineDoctorAction(action_id=11, target_column=outlier_column)]

    if observation.current_score >= TASK_THRESHOLDS[task_id] and not table_needs_attention(observation):
        return [PipelineDoctorAction(action_id=15), PipelineDoctorAction(action_id=14)]
    if policy_mode == "pure-llm":
        return [PipelineDoctorAction(action_id=14)]
    return [heuristic_action]


def task3_candidate_actions(
    observation: PipelineDoctorObservation,
    heuristic_action: PipelineDoctorAction,
    policy_mode: str,
) -> list[PipelineDoctorAction]:
    if policy_mode == "hybrid":
        return [heuristic_action]

    actions: list[PipelineDoctorAction] = []
    error_text = " | ".join(observation.recent_errors).lower()
    mismatch = first_schema_mismatch(observation)
    alias_action = alias_fix_action(observation)

    if policy_mode != "pure-llm" and heuristic_action.action_id != FALLBACK_ACTION.action_id:
        actions.append(heuristic_action)
    if alias_action:
        actions.append(alias_action)
    if observation.duplicate_rate > 0:
        actions.append(PipelineDoctorAction(action_id=10))

    null_column = column_from_errors(error_text, "null")
    if null_column:
        actions.extend(
            [
                PipelineDoctorAction(action_id=4, target_column=null_column),
                PipelineDoctorAction(action_id=5, target_column=null_column),
            ]
        )

    format_column = column_from_errors(error_text, "format mismatch")
    if observation.format_issues > 0 and format_column:
        actions.append(PipelineDoctorAction(action_id=9, target_column=format_column))
    if mismatch:
        column, info = mismatch
        actions.append(repair_action_for_mismatch(column, info))

    outlier_column = column_from_errors(error_text, "outlier")
    if observation.outlier_count > 0 and outlier_column:
        actions.append(PipelineDoctorAction(action_id=11, target_column=outlier_column))

    next_stage = next_table(observation.stage)
    if next_stage:
        actions.append(PipelineDoctorAction(action_id=0, target_column=next_stage))

    actions.append(PipelineDoctorAction(action_id=14))
    if observation.commit_ready or observation.current_score >= TASK_THRESHOLDS[3]:
        actions.append(PipelineDoctorAction(action_id=15))
    return dedupe_actions(actions) or [heuristic_action]


def task4_candidate_actions(
    observation: PipelineDoctorObservation,
    heuristic_action: PipelineDoctorAction,
    policy_mode: str,
) -> list[PipelineDoctorAction]:
    if policy_mode == "hybrid":
        return [heuristic_action]

    actions: list[PipelineDoctorAction] = []
    error_text = " | ".join(observation.recent_errors).lower()
    mismatch = first_schema_mismatch(observation)
    alias_action = alias_fix_action(observation)

    if policy_mode != "pure-llm" and heuristic_action.action_id != FALLBACK_ACTION.action_id:
        actions.append(heuristic_action)
    if alias_action:
        actions.append(alias_action)

    if observation.stage == "orders":
        if observation.backlog_rows > 0:
            if observation.resource_level < observation.required_resource_level:
                actions.append(PipelineDoctorAction(action_id=16))
            actions.append(PipelineDoctorAction(action_id=18))
        null_column = column_from_errors(error_text, "null")
        if null_column:
            actions.extend(
                [
                    PipelineDoctorAction(action_id=4, target_column=null_column),
                    PipelineDoctorAction(action_id=5, target_column=null_column),
                ]
            )

    if observation.duplicate_rate > 0:
        actions.append(PipelineDoctorAction(action_id=10))
    format_column = column_from_errors(error_text, "format mismatch")
    if observation.format_issues > 0 and format_column:
        actions.append(PipelineDoctorAction(action_id=9, target_column=format_column))
    if mismatch:
        column, info = mismatch
        actions.append(repair_action_for_mismatch(column, info))

    outlier_column = column_from_errors(error_text, "outlier")
    if observation.outlier_count > 0 and outlier_column:
        actions.append(PipelineDoctorAction(action_id=11, target_column=outlier_column))

    if observation.stage != "daily_summary":
        next_stage = next_table(observation.stage, task_id=4)
        if next_stage:
            actions.append(PipelineDoctorAction(action_id=0, target_column=next_stage))
    if observation.stage == "daily_summary":
        actions.append(PipelineDoctorAction(action_id=19))

    actions.append(PipelineDoctorAction(action_id=14))
    if observation.commit_ready or observation.current_score >= TASK_THRESHOLDS[4]:
        actions.append(PipelineDoctorAction(action_id=15))
    return dedupe_actions(actions) or [heuristic_action]


def task5_candidate_actions(
    observation: PipelineDoctorObservation,
    heuristic_action: PipelineDoctorAction,
    policy_mode: str,
) -> list[PipelineDoctorAction]:
    if policy_mode == "hybrid":
        return [heuristic_action]

    actions: list[PipelineDoctorAction] = []
    error_text = " | ".join(observation.recent_errors).lower()
    mismatch = first_schema_mismatch(observation)
    alias_action = alias_fix_action(observation)

    if policy_mode != "pure-llm" and heuristic_action.action_id != FALLBACK_ACTION.action_id:
        actions.append(heuristic_action)
    if alias_action:
        actions.append(alias_action)

    if observation.stage == "source_orders":
        if observation.backlog_rows > 0:
            if observation.resource_level < observation.required_resource_level:
                actions.append(PipelineDoctorAction(action_id=16))
            actions.append(PipelineDoctorAction(action_id=18))
        null_column = column_from_errors(error_text, "null")
        if null_column:
            actions.extend(
                [
                    PipelineDoctorAction(action_id=4, target_column=null_column),
                    PipelineDoctorAction(action_id=5, target_column=null_column),
                ]
            )

    if observation.duplicate_rate > 0:
        actions.append(PipelineDoctorAction(action_id=10))
    format_column = column_from_errors(error_text, "format mismatch")
    if observation.format_issues > 0 and format_column:
        actions.append(PipelineDoctorAction(action_id=9, target_column=format_column))
    if mismatch:
        column, info = mismatch
        actions.append(repair_action_for_mismatch(column, info))

    outlier_column = column_from_errors(error_text, "outlier")
    if observation.outlier_count > 0 and outlier_column:
        actions.append(PipelineDoctorAction(action_id=11, target_column=outlier_column))

    if observation.stage != "hourly_rollup":
        next_stage = next_table(observation.stage, task_id=5)
        if next_stage:
            actions.append(PipelineDoctorAction(action_id=0, target_column=next_stage))
    if observation.stage == "hourly_rollup":
        actions.append(PipelineDoctorAction(action_id=19))
        if observation.resource_level > observation.required_resource_level:
            actions.append(PipelineDoctorAction(action_id=17))

    actions.append(PipelineDoctorAction(action_id=14))
    if observation.commit_ready or observation.current_score >= TASK_THRESHOLDS[5]:
        actions.append(PipelineDoctorAction(action_id=15))
    return dedupe_actions(actions) or [heuristic_action]


def normalize_candidate_action(
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


def is_candidate_action(
    model_action: PipelineDoctorAction,
    candidate_actions: list[PipelineDoctorAction],
) -> bool:
    model_payload = model_action.model_dump(exclude_none=True)
    return any(
        candidate.model_dump(exclude_none=True) == model_payload
        for candidate in candidate_actions
    )


def task3_heuristic_action(
    observation: PipelineDoctorObservation,
) -> PipelineDoctorAction:
    error_text = " | ".join(observation.recent_errors).lower()
    mismatch = first_schema_mismatch(observation)
    alias_action = alias_fix_action(observation)
    if alias_action:
        return alias_action

    if observation.stage == "orders":
        if observation.duplicate_rate > 0:
            return PipelineDoctorAction(action_id=10)
        format_column = column_from_errors(error_text, "format mismatch")
        if format_column:
            return PipelineDoctorAction(action_id=9, target_column=format_column)
        if mismatch:
            column, info = mismatch
            return repair_action_for_mismatch(column, info)
        if only_calculation_mismatch(observation):
            return PipelineDoctorAction(action_id=0, target_column="customers")

    elif observation.stage == "customers":
        null_column = column_from_errors(error_text, "null")
        if null_column:
            return PipelineDoctorAction(action_id=4, target_column=null_column)
        format_column = column_from_errors(error_text, "format mismatch")
        if format_column:
            return PipelineDoctorAction(action_id=9, target_column=format_column)
        if mismatch:
            column, info = mismatch
            return repair_action_for_mismatch(column, info)
        if not table_needs_attention(observation):
            return PipelineDoctorAction(action_id=0, target_column="products")

    elif observation.stage == "products":
        if observation.duplicate_rate > 0:
            return PipelineDoctorAction(action_id=10)
        format_column = column_from_errors(error_text, "format mismatch")
        if format_column:
            return PipelineDoctorAction(action_id=9, target_column=format_column)
        if observation.outlier_count > 0:
            return PipelineDoctorAction(action_id=11, target_column="unit_price")
        if mismatch:
            column, info = mismatch
            return repair_action_for_mismatch(column, info)
        if not table_needs_attention(observation):
            return PipelineDoctorAction(action_id=15)
    return FALLBACK_ACTION


def task4_heuristic_action(
    observation: PipelineDoctorObservation,
) -> PipelineDoctorAction:
    error_text = " | ".join(observation.recent_errors).lower()
    mismatch = first_schema_mismatch(observation)
    alias_action = alias_fix_action(observation)
    if alias_action:
        return alias_action

    if observation.stage == "orders":
        if observation.backlog_rows > 0:
            if observation.resource_level < observation.required_resource_level:
                return PipelineDoctorAction(action_id=16)
            return PipelineDoctorAction(action_id=18)
        if observation.duplicate_rate > 0:
            return PipelineDoctorAction(action_id=10)
        format_column = column_from_errors(error_text, "format mismatch")
        if format_column:
            return PipelineDoctorAction(action_id=9, target_column=format_column)
        if mismatch:
            column, info = mismatch
            return repair_action_for_mismatch(column, info)
        if not table_needs_attention(observation):
            return PipelineDoctorAction(action_id=0, target_column="products")

    elif observation.stage == "products":
        if observation.duplicate_rate > 0:
            return PipelineDoctorAction(action_id=10)
        format_column = column_from_errors(error_text, "format mismatch")
        if format_column:
            return PipelineDoctorAction(action_id=9, target_column=format_column)
        if observation.outlier_count > 0:
            outlier_column = column_from_errors(error_text, "outlier") or "unit_price"
            return PipelineDoctorAction(action_id=11, target_column=outlier_column)
        if mismatch:
            column, info = mismatch
            return repair_action_for_mismatch(column, info)
        if not table_needs_attention(observation):
            return PipelineDoctorAction(action_id=0, target_column="daily_summary")

    elif observation.stage == "daily_summary":
        format_column = column_from_errors(error_text, "format mismatch")
        if format_column:
            return PipelineDoctorAction(action_id=9, target_column=format_column)
        if mismatch:
            column, info = mismatch
            return repair_action_for_mismatch(column, info)
        if observation.downstream_stale or observation.freshness_lag_minutes > 0:
            return PipelineDoctorAction(action_id=19)
        if observation.resource_level > observation.required_resource_level:
            return PipelineDoctorAction(action_id=17)
        if observation.commit_ready:
            return PipelineDoctorAction(action_id=15)

    if observation.commit_ready:
        return PipelineDoctorAction(action_id=15)
    return FALLBACK_ACTION


def task5_heuristic_action(
    observation: PipelineDoctorObservation,
) -> PipelineDoctorAction:
    error_text = " | ".join(observation.recent_errors).lower()
    mismatch = first_schema_mismatch(observation)
    alias_action = alias_fix_action(observation)
    if alias_action:
        return alias_action

    if observation.stage == "source_orders":
        if observation.backlog_rows > 0:
            if observation.resource_level < observation.required_resource_level:
                return PipelineDoctorAction(action_id=16)
            return PipelineDoctorAction(action_id=18)
        if observation.duplicate_rate > 0:
            return PipelineDoctorAction(action_id=10)
        format_column = column_from_errors(error_text, "format mismatch")
        if format_column:
            return PipelineDoctorAction(action_id=9, target_column=format_column)
        if mismatch:
            column, info = mismatch
            return repair_action_for_mismatch(column, info)
        if not table_needs_attention(observation):
            return PipelineDoctorAction(action_id=0, target_column="catalog")

    elif observation.stage == "catalog":
        if observation.duplicate_rate > 0:
            return PipelineDoctorAction(action_id=10)
        format_column = column_from_errors(error_text, "format mismatch")
        if format_column:
            return PipelineDoctorAction(action_id=9, target_column=format_column)
        if observation.outlier_count > 0:
            outlier_column = column_from_errors(error_text, "outlier") or "unit_price"
            return PipelineDoctorAction(action_id=11, target_column=outlier_column)
        if mismatch:
            column, info = mismatch
            return repair_action_for_mismatch(column, info)
        if not table_needs_attention(observation):
            return PipelineDoctorAction(action_id=0, target_column="hourly_rollup")

    elif observation.stage == "hourly_rollup":
        format_column = column_from_errors(error_text, "format mismatch")
        if format_column:
            return PipelineDoctorAction(action_id=9, target_column=format_column)
        if mismatch:
            column, info = mismatch
            return repair_action_for_mismatch(column, info)
        if observation.downstream_stale or observation.freshness_lag_minutes > 30:
            return PipelineDoctorAction(action_id=19)
        if observation.resource_level > observation.required_resource_level:
            return PipelineDoctorAction(action_id=17)
        if observation.commit_ready:
            return PipelineDoctorAction(action_id=15)

    if observation.commit_ready:
        return PipelineDoctorAction(action_id=15)
    return FALLBACK_ACTION


def dedupe_actions(actions: list[PipelineDoctorAction]) -> list[PipelineDoctorAction]:
    deduped: list[PipelineDoctorAction] = []
    seen: set[str] = set()
    for action in actions:
        key = json.dumps(action.model_dump(exclude_none=True), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(action)
    return deduped


def only_calculation_mismatch(observation: PipelineDoctorObservation) -> bool:
    return bool(
        observation.stage == "orders"
        and observation.missing_rate == 0
        and observation.duplicate_rate == 0
        and observation.type_violations == 0
        and observation.outlier_count == 0
        and observation.format_issues == 0
        and any("calculation mismatch" in error.lower() for error in observation.recent_errors)
    )
