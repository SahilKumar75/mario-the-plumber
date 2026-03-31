"""Policy-mode orchestration for Mario baseline runs."""

from __future__ import annotations

from openai import OpenAI

try:
    from ...models import PipelineDoctorAction, PipelineDoctorObservation
except ImportError:
    from models import PipelineDoctorAction, PipelineDoctorObservation

try:
    from .candidates import (
        candidate_actions_for,
        is_candidate_action,
        normalize_candidate_action,
    )
    from .heuristics import FALLBACK_ACTION, heuristic_action_for
    from .prompts import SYSTEM_PROMPT, build_user_prompt, parse_action
    from .utils import (
        action_has_required_fields,
        same_action,
        table_needs_attention,
    )
except ImportError:
    from benchmark.policies.candidates import (
        candidate_actions_for,
        is_candidate_action,
        normalize_candidate_action,
    )
    from benchmark.policies.heuristics import FALLBACK_ACTION, heuristic_action_for
    from benchmark.policies.prompts import SYSTEM_PROMPT, build_user_prompt, parse_action
    from benchmark.policies.utils import (
        action_has_required_fields,
        same_action,
        table_needs_attention,
    )

TEMPERATURE = 0.0
MAX_TOKENS = 220
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
