from __future__ import annotations

from benchmark.api_payloads import tasks_payload
from benchmark.grading import score_task4
from benchmark.api_payloads import benchmark_metadata_payload
from models import PipelineDoctorAction
from server.data_generator import generate_scenario
from server.pipeline_doctor_environment import PipelineDoctorEnvironment


def test_reset_and_step_lifecycle_invariants() -> None:
    fresh_env = PipelineDoctorEnvironment()
    fresh_observation = fresh_env.reset(task_id=4, split="eval", seed=7)

    assert fresh_env.state.step_count == 0
    assert fresh_observation.steps_taken == 0
    assert fresh_observation.time_budget_remaining == fresh_env.state.max_steps
    assert fresh_observation.current_score == fresh_env.state.current_score
    assert fresh_env.state.initial_score == fresh_observation.current_score

    step_observation = fresh_env.step(PipelineDoctorAction(action_id=2))

    assert fresh_env.state.step_count == 1
    assert step_observation.steps_taken == 1
    assert step_observation.time_budget_remaining == fresh_env.state.max_steps - 1
    assert step_observation.done is False
    assert step_observation.done_reason == ""
    assert step_observation.current_score == fresh_observation.current_score

    dirty_env = PipelineDoctorEnvironment()
    dirty_env.reset(task_id=5, split="eval", seed=7)
    dirty_env.step(PipelineDoctorAction(action_id=16))
    dirty_observation = dirty_env.reset(task_id=4, split="eval", seed=7)

    assert dirty_observation.current_score == fresh_observation.current_score
    assert dirty_env.state.initial_score == fresh_observation.current_score


def test_task3_commit_requires_dependency_repair_and_action_19_restores_it() -> None:
    env = PipelineDoctorEnvironment()
    env.reset(task_id=3, split="eval", seed=7)

    env._tables = {
        name: frame.copy(deep=True)
        for name, frame in env._ground_truth.items()
    }
    env._tables["orders"]["total_price"] = env._tables["orders"]["total_price"] + 5.0
    env._refresh_errors()
    env._update_task_progress_state()

    assert env._task3_commit_ready() is False
    repair_observation = env.step(PipelineDoctorAction(action_id=19))
    assert repair_observation.dependency_alerts == []
    assert env._task3_commit_ready() is True

    commit_observation = env.step(PipelineDoctorAction(action_id=15))
    assert commit_observation.done is True
    assert commit_observation.current_score >= 0.75
    assert env.state.done is True
    assert env.state.success is True


def test_task4_commit_stays_blocked_while_incident_pressure_remains() -> None:
    env = PipelineDoctorEnvironment()
    env.reset(task_id=4, split="eval", seed=7)

    assert env._task4_commit_ready() is False

    for action_id in (16, 16, 18, 19):
        env.step(PipelineDoctorAction(action_id=action_id))

    assert env.state.backlog_rows == 0
    assert env.state.freshness_lag_minutes == 0
    assert env.state.resource_level == 3
    assert env._task4_commit_ready() is False

    commit_observation = env.step(PipelineDoctorAction(action_id=15))
    assert commit_observation.done is True
    assert commit_observation.done_reason == "commit_failure"
    assert env.state.success is False


def test_task5_commit_stays_blocked_until_temporal_recovery_is_complete() -> None:
    env = PipelineDoctorEnvironment()
    observation = env.reset(task_id=5, split="eval", seed=1)

    assert observation.scenario_profile.startswith("heldout_temporal_")
    assert env._task5_commit_ready() is False

    for action_id in (16, 16, 18, 19):
        env.step(PipelineDoctorAction(action_id=action_id))

    assert env.state.backlog_rows == 0
    assert env.state.freshness_lag_minutes > 0
    assert env.state.resource_level == 3
    assert env._task5_commit_ready() is False

    commit_observation = env.step(PipelineDoctorAction(action_id=15))
    assert commit_observation.done is True
    assert commit_observation.done_reason == "commit_failure"
    assert env.state.success is False


def test_observation_contract_includes_incident_and_recovery_fields() -> None:
    observation = PipelineDoctorEnvironment().reset(task_id=5, split="eval", seed=2)

    assert observation.incident_type
    assert observation.incident_summary
    assert observation.diagnosis_signals
    assert observation.recovery_requirements
    assert observation.unsafe_commit_conditions
    assert observation.recent_errors
    assert observation.commit_ready is False
    assert observation.available_actions == list(range(20))
    assert isinstance(observation.reward_breakdown, dict)
    assert observation.objective_breakdown
    assert observation.tradeoff_weights
    assert observation.subgoal_progress
    assert observation.reward_machine_state
    assert observation.active_subgoal
    assert observation.adaptation_target
    assert observation.dependency_alerts
    assert observation.dependency_health_summary
    assert observation.recent_failure_counters
    assert observation.drift_markers
    assert observation.queue_backlog_age_minutes > 0
    assert observation.freshness_lag_minutes > 0
    assert observation.sla_severity in {"high", "critical"}
    assert observation.time_budget_remaining == 35
    assert observation.heldout_profile_family is False


def test_task4_scoring_matches_published_weights() -> None:
    scenario = generate_scenario(task_id=4, split="eval", seed=7)
    perfect_tables = {
        name: frame.copy(deep=True)
        for name, frame in scenario.ground_truth_tables.items()
    }
    score, breakdown = score_task4(
        perfect_tables,
        scenario.ground_truth_tables,
        scenario.expected_types,
        backlog_rows=0,
        freshness_lag_minutes=0,
        resource_level=1,
        required_resource_level=1,
        downstream_stale=False,
    )

    assert score == 1.0
    assert breakdown["pipeline"] == {
        "data_quality": 1.0,
        "backlog": 1.0,
        "freshness": 1.0,
        "summary_consistency": 1.0,
        "resource_efficiency": 1.0,
    }


def test_benchmark_metadata_and_tasks_payload_smoke() -> None:
    metadata = benchmark_metadata_payload()
    payload = tasks_payload()

    assert metadata["benchmark_version"] == "2.1"
    assert metadata["runtime_mode"] == "benchmark"
    assert metadata["runtime_mode_card"]["summary"]
    assert metadata["task_names"][5] == "Temporal Rollup Recovery"
    assert metadata["formal_task_specs"][5]["reward_machine_order"]
    assert metadata["objective_weights"][5]["rollup_consistency"] == 0.2

    assert len(payload["tasks"]) == 5
    assert "incident_signals" in payload["action_schema"]
    assert "reward_machine_signals" in payload["action_schema"]
    assert "incident_type" in payload["action_schema"]["incident_signals"]
    assert "reward_machine_state" in payload["action_schema"]["reward_machine_signals"]
