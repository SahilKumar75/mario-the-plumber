"""Deterministic grader helpers for the validator-facing Mario task registry."""

from __future__ import annotations

from benchmark.evaluation import breakdown_payload, objective_breakdown
from benchmark.policies.heuristics import heuristic_action_for
from benchmark.task_ids import parse_task_id, public_task_id
from models import PipelineDoctorAction
from server.pipeline_doctor_environment import EPISODE_SUMMARIES, PipelineDoctorEnvironment
from tasks.task_bank import get_task


def _jsonify(value):
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _normalize_stored_payload(task_id: int, episode_id: str, payload: dict[str, object]) -> dict[str, object]:
    task = get_task(public_task_id(task_id))
    score = round(float(payload.get("score", 0.0)), 4)
    success = bool(payload.get("success", False))
    return {
        "task_id": task.internal_id,
        "task_alias": task.id,
        "episode_id": episode_id,
        "score": score,
        "reward": score,
        "success_threshold": task.success_threshold,
        "success": success,
        "breakdown": _jsonify(payload.get("breakdown", {})),
        "objective_breakdown": _jsonify(payload.get("objective_breakdown", {})),
        "grader_mode": "stored",
        "steps_taken": int(payload.get("steps_taken", 0)),
        "truncated": bool(payload.get("truncated", False)),
        "done_reason": payload.get("done_reason", ""),
    }


def grade_env(env: PipelineDoctorEnvironment, *, grader_mode: str, episode_id: str | None = None) -> dict[str, object]:
    task = get_task(public_task_id(env.state.task_id))
    score = round(float(env.state.current_score), 4)
    return {
        "task_id": task.internal_id,
        "task_alias": task.id,
        "episode_id": episode_id or env.state.episode_id,
        "score": score,
        "reward": score,
        "success_threshold": task.success_threshold,
        "success": bool(env.state.success),
        "breakdown": _jsonify(breakdown_payload(env)),
        "objective_breakdown": _jsonify(objective_breakdown(env)),
        "grader_mode": grader_mode,
        "steps_taken": env.state.step_count,
        "truncated": bool(env.state.truncated),
        "done_reason": env.state.done_reason,
    }


def run_live_grade(
    task_ref: int | str,
    *,
    seed: int = 42,
    split: str = "eval",
    episode_id: str | None = None,
) -> dict[str, object]:
    """Run a deterministic heuristic episode to completion and return its grade."""

    task_id = parse_task_id(task_ref)
    env = PipelineDoctorEnvironment()
    observation = env.reset(seed=seed, task_id=task_id, split=split, episode_id=episode_id)

    while not env.state.done:
        observation = env.step(heuristic_action_for(task_id, observation))

    if not env.state.done:
        env.step(PipelineDoctorAction(action_id=15))

    return grade_env(env, grader_mode="live", episode_id=episode_id)


def grade_episode(
    task_ref: int | str,
    *,
    episode_id: str | None = None,
    seed: int = 42,
    split: str = "eval",
) -> dict[str, object]:
    """Grade a stored episode when available, else fall back to deterministic live grading."""

    task_id = parse_task_id(task_ref)
    if episode_id:
        payload = EPISODE_SUMMARIES.get(episode_id)
        if payload is not None:
            return _normalize_stored_payload(task_id, episode_id, payload)
    result = run_live_grade(task_id, seed=seed, split=split, episode_id=episode_id)
    if episode_id and result.get("episode_id") is None:
        result["episode_id"] = episode_id
    return result
