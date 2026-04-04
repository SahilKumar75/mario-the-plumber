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
