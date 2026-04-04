from __future__ import annotations

try:
    from .tasks.shared import duplicate_row_count, score_single_table
    from .tasks.task3 import calculation_mismatch_count, score_task3, task3_dependency_score
    from .tasks.task4 import (
        score_task4,
        task4_batch_completeness_score,
        task4_summary_consistency_score,
    )
    from .tasks.task5 import score_task5, task5_rollup_consistency_score
except ImportError:
    from benchmark.tasks.shared import duplicate_row_count, score_single_table
    from benchmark.tasks.task3 import calculation_mismatch_count, score_task3, task3_dependency_score
    from benchmark.tasks.task4 import (
        score_task4,
        task4_batch_completeness_score,
        task4_summary_consistency_score,
    )
    from benchmark.tasks.task5 import score_task5, task5_rollup_consistency_score

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
    "task5_rollup_consistency_score",
]


def compute_reward(
    score_before: float,
    score_after: float,
    *,
    action_valid: bool,
    done: bool,
    success: bool,
) -> float:
    reward = 0.5 * (score_after - score_before)
    reward -= 0.001
    if not action_valid:
        reward -= 0.05
    if done and success:
        reward += 1.0
    if done and not success:
        reward -= 0.5
    return round(reward, 4)


def compute_reward_breakdown(
    score_before: float,
    score_after: float,
    *,
    action_valid: bool,
    done: bool,
    success: bool,
) -> dict[str, float]:
    progress = round(0.5 * (score_after - score_before), 4)
    step_cost = -0.001
    invalid_penalty = -0.05 if not action_valid else 0.0
    terminal_bonus = 1.0 if done and success else 0.0
    terminal_penalty = -0.5 if done and not success else 0.0
    total = round(progress + step_cost + invalid_penalty + terminal_bonus + terminal_penalty, 4)
    return {
        "progress": progress,
        "step_cost": step_cost,
        "invalid_penalty": invalid_penalty,
        "terminal_bonus": terminal_bonus,
        "terminal_penalty": terminal_penalty,
        "total": total,
    }
