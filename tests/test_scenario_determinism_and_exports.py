from __future__ import annotations

import pytest
from pandas.testing import assert_frame_equal

from scripts.export_benchmark_metadata import collect_initial_score_stats
from server.data_generator import generate_scenario


def _assert_scenario_tables_equal(left, right) -> None:
    assert left.broken_tables.keys() == right.broken_tables.keys()
    assert left.ground_truth_tables.keys() == right.ground_truth_tables.keys()
    for table_name in left.broken_tables:
        assert_frame_equal(left.broken_tables[table_name], right.broken_tables[table_name])
        assert_frame_equal(left.ground_truth_tables[table_name], right.ground_truth_tables[table_name])


@pytest.mark.parametrize("task_id", [3, 4, 5])
def test_generate_scenario_is_deterministic_by_task_split_seed(task_id: int) -> None:
    eval_first = generate_scenario(task_id=task_id, split="eval", seed=7)
    eval_second = generate_scenario(task_id=task_id, split="eval", seed=7)
    train_variant = generate_scenario(task_id=task_id, split="train", seed=7)

    _assert_scenario_tables_equal(eval_first, eval_second)
    assert eval_first.seed == eval_second.seed == 7
    assert eval_first.split == eval_second.split == "eval"
    assert eval_first.active_table == eval_second.active_table
    assert eval_first.metadata["scenario_profile"] == eval_second.metadata["scenario_profile"]
    assert eval_first.metadata.keys() == eval_second.metadata.keys()
    assert train_variant.metadata["scenario_profile"] != eval_first.metadata["scenario_profile"]
    assert any(
        not train_variant.broken_tables[name].equals(eval_first.broken_tables[name])
        for name in eval_first.broken_tables
    )


def test_task5_train_and_eval_profiles_follow_heldout_split_semantics() -> None:
    train = generate_scenario(task_id=5, split="train", seed=7)
    eval_ = generate_scenario(task_id=5, split="eval", seed=7)

    assert not train.metadata["scenario_profile"].startswith("heldout_temporal_")
    assert eval_.metadata["scenario_profile"].startswith("heldout_temporal_")
    assert train.metadata["scenario_profile"] != eval_.metadata["scenario_profile"]
    assert train.metadata["incident_manifest"]["dag_id"] == "temporal_orders_rollup"
    assert eval_.metadata["incident_manifest"]["dag_id"] == "temporal_orders_rollup"
    assert train.metadata["incident_manifest"]["expected_watermark_after_replay"]
    assert eval_.metadata["incident_manifest"]["affected_hour_buckets"]
    assert eval_.metadata["incident_manifest"]["novelty_axes"]
    assert train.metadata["operational_trace_summary"]
    assert eval_.metadata["operational_trace_summary"]


def test_collect_initial_score_stats_reports_split_and_task_coverage() -> None:
    stats = collect_initial_score_stats([1, 2])

    assert set(stats) == {"train", "eval"}
    assert set(stats["train"]) == {f"task_{task_id}" for task_id in range(1, 6)}
    assert set(stats["eval"]) == {f"task_{task_id}" for task_id in range(1, 6)}

    train_task5 = stats["train"]["task_5"]
    eval_task5 = stats["eval"]["task_5"]

    assert train_task5["threshold"] == 0.82
    assert eval_task5["threshold"] == 0.82
    assert train_task5["profiles_seen"] == {
        "schema_evolution_backfill_recovery": 1,
        "late_correction_backpressure_incident": 1,
    }
    assert set(eval_task5["profiles_seen"]) == {
        "heldout_temporal_schema_extension_family",
        "late_correction_backpressure_incident",
    }
    assert eval_task5["initial_score_mean"] >= 0.0
