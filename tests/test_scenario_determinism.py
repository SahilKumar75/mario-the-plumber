from __future__ import annotations

from server.data_generator import generate_scenario


def test_trace_grounded_scenarios_are_deterministic_for_same_seed() -> None:
    for task_id in (3, 4, 5):
        first = generate_scenario(task_id=task_id, split="eval", seed=11)
        second = generate_scenario(task_id=task_id, split="eval", seed=11)
        assert first.metadata["scenario_profile"] == second.metadata["scenario_profile"]
        assert first.metadata["incident_manifest"] == second.metadata["incident_manifest"]
        assert first.metadata["warehouse_events"].equals(second.metadata["warehouse_events"])
        assert first.metadata["dag_runs"].equals(second.metadata["dag_runs"])
        for table_name in first.broken_tables:
            assert first.broken_tables[table_name].equals(second.broken_tables[table_name])
            assert first.ground_truth_tables[table_name].equals(second.ground_truth_tables[table_name])
