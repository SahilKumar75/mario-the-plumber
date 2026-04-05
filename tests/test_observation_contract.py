from __future__ import annotations

from server.pipeline_doctor_environment import PipelineDoctorEnvironment


def test_task5_observation_exposes_trace_grounded_contract() -> None:
    env = PipelineDoctorEnvironment()
    observation = env.reset(task_id=5, split="eval", seed=2)

    assert observation.incident_type
    assert observation.incident_summary
    assert observation.diagnosis_signals
    assert observation.recovery_requirements
    assert observation.unsafe_commit_conditions
    assert observation.recent_failure_counters
    assert observation.drift_markers
    assert observation.dependency_health_summary
    assert observation.open_world_patterns
    assert isinstance(observation.synthetic_data_notes, list)
    assert observation.adaptation_target
    assert isinstance(observation.heldout_profile_family, bool)


def test_task1_and_task2_observation_trim_hard_task_bundles() -> None:
    env = PipelineDoctorEnvironment()

    for task_id in (1, 2):
        observation = env.reset(task_id=task_id, split="eval", seed=2)

        assert observation.tradeoff_weights == {}
        assert observation.subgoal_progress == {}
        assert observation.subgoal_order == []
        assert observation.active_subgoal == ""
        assert observation.reward_machine_state == ""
        assert observation.recent_failure_counters == {}
        assert observation.drift_markers == []
        assert observation.dependency_health_summary == {}
        assert observation.adaptation_target == ""
        assert observation.repeated_action_streak == 0
        assert observation.repeated_action_tripwire is False
        assert observation.incident_type
        assert observation.incident_summary
        assert observation.observed_columns


def test_task3_to_task5_observation_keep_structured_recovery_fields() -> None:
    env = PipelineDoctorEnvironment()

    for task_id in (3, 4, 5):
        observation = env.reset(task_id=task_id, split="eval", seed=2)

        assert observation.tradeoff_weights
        assert observation.subgoal_progress
        assert observation.subgoal_order
        assert observation.reward_machine_state
        assert observation.recent_failure_counters
        assert observation.drift_markers
        assert observation.dependency_health_summary
        assert observation.adaptation_target
        assert observation.repeated_action_streak == 0
        assert observation.repeated_action_tripwire is False
