"""Compatibility re-export for Mario grading helpers."""

from __future__ import annotations

from benchmark.grading import (
    calculation_mismatch_count,
    compute_reward,
    compute_reward_breakdown,
    duplicate_row_count,
    score_single_table,
    score_task3,
    score_task4,
    score_task5,
    task3_dependency_score,
    task4_batch_completeness_score,
    task4_summary_consistency_score,
    task5_rollup_consistency_score,
)

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
