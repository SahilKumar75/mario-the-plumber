from __future__ import annotations

from models import PipelineDoctorAction
from server.pipeline_doctor_environment import PipelineDoctorEnvironment


def test_reset_recomputes_score_from_fresh_state() -> None:
    env = PipelineDoctorEnvironment()

    first = env.reset(seed=7, task_id=4, split="eval")
    env.step(PipelineDoctorAction(action_id=0))
    env.step(PipelineDoctorAction(action_id=1))
    env.step(PipelineDoctorAction(action_id=2))
    second = env.reset(seed=7, task_id=4, split="eval")

    assert first.current_score == second.current_score


def test_task3_commit_requires_dependency_consistency() -> None:
    env = PipelineDoctorEnvironment()
    env.reset(seed=1, task_id=3, split="train")
    env._tables = {name: frame.copy(deep=True) for name, frame in env._ground_truth.items()}
    env._tables["orders"]["total_price"] = 0.0

    pre_commit = env._build_observation(reward=0.0, done=False)
    assert pre_commit.commit_ready is False

    failed_commit = env.step(PipelineDoctorAction(action_id=15))
    assert env.state.success is False
    assert failed_commit.current_score < 1.0


def test_task3_refresh_action_repairs_dependency_before_commit() -> None:
    env = PipelineDoctorEnvironment()
    env.reset(seed=1, task_id=3, split="train")
    env._tables = {name: frame.copy(deep=True) for name, frame in env._ground_truth.items()}
    env._tables["orders"]["total_price"] = 0.0

    refresh = env.step(PipelineDoctorAction(action_id=19))
    assert "refreshed" in refresh.action_result.lower()
    assert env._build_observation(reward=0.0, done=False).commit_ready is True

    committed = env.step(PipelineDoctorAction(action_id=15))
    assert env.state.success is True
    assert committed.current_score == 1.0


def test_task4_perfect_recovered_state_scores_one() -> None:
    env = PipelineDoctorEnvironment()
    env.reset(seed=1, task_id=4, split="eval")
    env._tables = {name: frame.copy(deep=True) for name, frame in env._ground_truth.items()}
    env._state.backlog_rows = 0
    env._state.pending_batches = 0
    env._state.queue_backlog_age_minutes = 0
    env._state.freshness_lag_minutes = 0
    env._state.resource_level = env._state.required_resource_level
    env._scenario_meta["backlog_rows"] = 0
    env._scenario_meta["pending_batches"] = 0
    env._scenario_meta["queue_backlog_age_minutes"] = 0
    env._scenario_meta["freshness_lag_minutes"] = 0
    env._scenario_meta["downstream_stale"] = False

    assert env._score() == 1.0
