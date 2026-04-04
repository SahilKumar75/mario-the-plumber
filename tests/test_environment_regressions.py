from __future__ import annotations

from benchmark.grading import score_task4
from models import PipelineDoctorAction
from server.data_generator import generate_scenario
from server.pipeline_doctor_environment import PipelineDoctorEnvironment


def test_task4_reset_score_uses_new_episode_state() -> None:
    fresh_env = PipelineDoctorEnvironment()
    fresh_observation = fresh_env.reset(task_id=4, split="eval", seed=7)

    dirty_env = PipelineDoctorEnvironment()
    dirty_env.reset(task_id=5, split="eval", seed=7)
    dirty_observation = dirty_env.reset(task_id=4, split="eval", seed=7)

    assert dirty_observation.current_score == fresh_observation.current_score
    assert dirty_env.state.initial_score == fresh_env.state.initial_score


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
