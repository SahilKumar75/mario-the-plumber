from __future__ import annotations

from benchmark.tasks.shared import duplicate_row_count, score_single_table
from benchmark.tasks.task3 import calculation_mismatch_count, score_task3, task3_dependency_score
from benchmark.tasks.task4 import (
    score_task4,
    task4_batch_completeness_score,
    task4_summary_consistency_score,
)
from benchmark.tasks.task5 import score_task5, task5_rollup_consistency_score, task5_temporal_closure_score

__all__ = [
    "calculation_mismatch_count",
    "compute_reward",
    "compute_reward_breakdown",
    "duplicate_row_count",
    "score_single_table",
    "score_task3",
    "score_task4",
    "score_task5",
    "task3_dependency_score",
    "task4_batch_completeness_score",
    "task4_summary_consistency_score",
    "task5_temporal_closure_score",
    "task5_rollup_consistency_score",
]

PROGRESS_WEIGHT = 0.9
STEP_COST = -0.004
INVALID_PENALTY = -0.06
TERMINAL_SUCCESS_BASE = 0.18
TERMINAL_SUCCESS_PROGRESS_CAP = 0.1
TERMINAL_FAILURE_PENALTY = -0.45
PREMATURE_COMMIT_PENALTY = -0.25


def _terminal_bonus(score_before: float, score_after: float, *, done: bool, success: bool) -> float:
    if not done or not success:
        return 0.0
    progress_component = max(score_after - score_before, 0.0)
    return round(TERMINAL_SUCCESS_BASE + min(progress_component, TERMINAL_SUCCESS_PROGRESS_CAP), 4)


def compute_reward(
    score_before: float,
    score_after: float,
    *,
    action_valid: bool,
    done: bool,
    success: bool,
    action_id: int = -1,
    task_threshold: float = 1.0,
) -> float:
    reward = PROGRESS_WEIGHT * (score_after - score_before)
    reward += STEP_COST
    if not action_valid:
        reward += INVALID_PENALTY
    if action_id == 15 and score_before < task_threshold:
        reward += PREMATURE_COMMIT_PENALTY
    if done and success:
        reward += _terminal_bonus(score_before, score_after, done=done, success=success)
    if done and not success:
        reward += TERMINAL_FAILURE_PENALTY
    return round(reward, 4)


def compute_reward_breakdown(
    score_before: float,
    score_after: float,
    *,
    action_valid: bool,
    done: bool,
    success: bool,
    action_id: int = -1,
    task_threshold: float = 1.0,
) -> dict[str, float]:
    progress = round(PROGRESS_WEIGHT * (score_after - score_before), 4)
    step_cost = STEP_COST
    invalid_penalty = INVALID_PENALTY if not action_valid else 0.0
    premature_commit_penalty = PREMATURE_COMMIT_PENALTY if action_id == 15 and score_before < task_threshold else 0.0
    terminal_bonus = _terminal_bonus(score_before, score_after, done=done, success=success)
    terminal_penalty = TERMINAL_FAILURE_PENALTY if done and not success else 0.0
    total = round(progress + step_cost + invalid_penalty + premature_commit_penalty + terminal_bonus + terminal_penalty, 4)
    return {
        "progress": progress,
        "step_cost": step_cost,
        "invalid_penalty": invalid_penalty,
        "premature_commit_penalty": premature_commit_penalty,
        "terminal_bonus": terminal_bonus,
        "terminal_penalty": terminal_penalty,
        "total": total,
    }
