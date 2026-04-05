from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

import pandas as pd

from benchmark.action_metadata import ACTION_NAMES
from benchmark.catalog import MAX_STEPS, TASK_THRESHOLDS
from benchmark.grading import compute_reward, compute_reward_breakdown
from models import PipelineDoctorState


@dataclass(slots=True)
class StepResolution:
    action_name: str
    reward: float
    done: bool
    success: bool
    truncated: bool
    done_reason: str
    score_after: float
    reward_breakdown: dict[str, float]


def initialize_episode(env, scenario, *, task_id: int, seed: int | None, episode_id: str | None) -> None:
    env._tables = {name: frame.copy(deep=True) for name, frame in scenario.broken_tables.items()}
    env._ground_truth = {
        name: frame.copy(deep=True) for name, frame in scenario.ground_truth_tables.items()
    }
    env._expected_types = dict(scenario.expected_types)
    env._scenario_meta = {
        key: _copy_metadata_value(value) for key, value in scenario.metadata.items()
    }
    env._task_id = task_id
    env._seed = seed
    env._split = scenario.split
    env._recent_errors = []
    env._state = PipelineDoctorState(
        episode_id=episode_id or str(uuid4()),
        task_id=task_id,
        seed=seed,
        step_count=0,
        max_steps=MAX_STEPS[task_id],
        current_score=0.0,
        initial_score=0.0,
        best_score=0.0,
        done=False,
        success=None,
        active_table=scenario.active_table,
        scenario_split=scenario.split,
        backlog_rows=int(env._scenario_meta.get("backlog_rows", 0)),
        queue_backlog_age_minutes=int(env._scenario_meta.get("queue_backlog_age_minutes", 0)),
        freshness_lag_minutes=int(env._scenario_meta.get("freshness_lag_minutes", 0)),
        sla_severity="none",
        resource_level=int(env._scenario_meta.get("resource_level", 1)),
        required_resource_level=int(env._scenario_meta.get("required_resource_level", 1)),
        pending_batches=int(env._scenario_meta.get("pending_batches", 0)),
        time_budget_remaining=MAX_STEPS[task_id],
        truncated=False,
        done_reason="",
        scenario_profile=str(env._scenario_meta.get("scenario_profile", "baseline")),
        started_at=datetime.now(UTC).isoformat(),
        active_subgoal="",
        reward_machine_state="",
        heldout_profile_family=bool(env._scenario_meta.get("heldout_profile_family", False)),
    )
    env._refresh_errors()
    env._update_task_progress_state()
    current_score = env._score()
    env._state.current_score = current_score
    env._state.initial_score = current_score
    env._state.best_score = current_score


def resolve_step(env, *, action_id: int, score_before: float, action_valid: bool) -> StepResolution:
    env._refresh_errors()
    commit_ready = env._commit_ready()
    score_after = env._score()

    threshold = TASK_THRESHOLDS[env._task_id]
    done = False
    success = False
    truncated = False
    done_reason = ""
    if action_id == 15:
        done = True
        success = commit_ready and score_after >= threshold and action_valid
        done_reason = "commit_success" if success else "commit_failure"
    elif score_after < 0.10:
        done = True
        done_reason = "quality_collapse"
    elif env._state.step_count >= env._state.max_steps:
        done = True
        truncated = True
        done_reason = "step_budget_exhausted"

    reward = compute_reward(
        score_before,
        score_after,
        action_valid=action_valid,
        done=done,
        success=success,
    )
    reward_breakdown = compute_reward_breakdown(
        score_before,
        score_after,
        action_valid=action_valid,
        done=done,
        success=success,
    )

    env._state.current_score = score_after
    env._state.best_score = max(env._state.best_score, score_after)
    env._state.done = done
    env._state.success = success if done else None
    env._state.truncated = truncated
    env._state.done_reason = done_reason
    env._state.time_budget_remaining = max(0, env._state.max_steps - env._state.step_count)
    env._update_task_progress_state()

    return StepResolution(
        action_name=ACTION_NAMES.get(action_id, "unknown"),
        reward=reward,
        done=done,
        success=success,
        truncated=truncated,
        done_reason=done_reason,
        score_after=score_after,
        reward_breakdown=reward_breakdown,
    )


def _copy_metadata_value(value: object) -> object:
    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)
    if isinstance(value, dict):
        return {key: _copy_metadata_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_metadata_value(item) for item in value]
    return value
