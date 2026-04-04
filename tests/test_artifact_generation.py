from __future__ import annotations

import json

import pytest

pytest.importorskip("matplotlib")

from scripts import generate_visuals


def test_generate_visuals_smoke(tmp_path) -> None:
    generate_visuals.ASSETS = tmp_path
    (tmp_path / "benchmark_runs.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"policy": "random", "split": "train", "task_1": 0.1, "task_2": 0.2, "task_3": 0.3, "task_4": 0.4, "task_5": 0.5},
                    {"policy": "heuristic", "split": "train", "task_1": 0.8, "task_2": 0.8, "task_3": 0.8, "task_4": 0.8, "task_5": 0.8},
                    {"policy": "random", "split": "eval", "task_1": 0.1, "task_2": 0.2, "task_3": 0.3, "task_4": 0.4, "task_5": 0.5},
                    {"policy": "heuristic", "split": "eval", "task_1": 0.7, "task_2": 0.7, "task_3": 0.7, "task_4": 0.7, "task_5": 0.7},
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "benchmark_metadata.json").write_text(
        json.dumps(
            {
                "benchmark_metadata": {
                    "objective_weights": {
                        "1": {"accuracy": 0.3, "completeness": 0.2, "consistency": 0.3, "validity": 0.2},
                        "2": {"accuracy": 0.3, "completeness": 0.2, "consistency": 0.3, "validity": 0.2},
                        "3": {"data_quality": 0.55, "dependency_consistency": 0.45},
                        "4": {"data_quality": 0.45, "freshness": 0.2, "backlog": 0.15, "resource_efficiency": 0.1, "summary_consistency": 0.1},
                        "5": {"schema_alignment": 0.2, "temporal_backfill": 0.2, "rollup_consistency": 0.2, "freshness": 0.15, "resource_efficiency": 0.1, "data_quality": 0.15},
                    }
                },
                "initial_score_stats": {
                    "train": {f"task_{task_id}": {"initial_score_mean": 0.2} for task_id in range(1, 6)},
                    "eval": {f"task_{task_id}": {"initial_score_mean": 0.25} for task_id in range(1, 6)},
                },
            }
        ),
        encoding="utf-8",
    )

    generate_visuals.main()

    assert (tmp_path / "benchmark_overview.png").exists()
    assert (tmp_path / "difficulty_gap.png").exists()
    assert (tmp_path / "objective_weights.png").exists()
