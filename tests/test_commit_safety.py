from __future__ import annotations

from models import PipelineDoctorAction
from server.pipeline_doctor_environment import PipelineDoctorEnvironment


def test_task4_commit_requires_operational_recovery() -> None:
    env = PipelineDoctorEnvironment()
    env.reset(task_id=4, split="eval", seed=2)
    env._tables = {
        name: frame.copy(deep=True)
        for name, frame in env._ground_truth.items()
    }
    env._refresh_errors()
    env._update_task_progress_state()

    assert env._commit_ready() is False
    failed_commit = env.step(PipelineDoctorAction(action_id=15))
    assert failed_commit.done is True
    assert failed_commit.done_reason == "commit_failure"


def test_task5_commit_requires_temporal_replay_and_refresh() -> None:
    env = PipelineDoctorEnvironment()
    env.reset(task_id=5, split="eval", seed=2)
    env._tables = {
        name: frame.copy(deep=True)
        for name, frame in env._ground_truth.items()
    }
    env._refresh_errors()
    env._update_task_progress_state()

    assert env._commit_ready() is False
    failed_commit = env.step(PipelineDoctorAction(action_id=15))
    assert failed_commit.done is True
    assert failed_commit.done_reason == "commit_failure"


def test_task4_prioritize_incremental_batch_replays_one_batch_at_a_time() -> None:
    env = PipelineDoctorEnvironment()
    env.reset(task_id=4, split="eval", seed=2)

    initial_backlog = env.state.backlog_rows
    initial_batches = env.state.pending_batches
    expected_first_batch = sorted(
        env._scenario_meta["pending_orders"]["batch_id"].astype(str).unique().tolist()
    )[0]

    env.step(PipelineDoctorAction(action_id=16))
    env.step(PipelineDoctorAction(action_id=16))
    replay = env.step(PipelineDoctorAction(action_id=18))

    assert replay.done is False
    assert env.state.backlog_rows < initial_backlog
    assert env.state.backlog_rows > 0
    assert env.state.pending_batches == initial_batches - 1
    assert env._scenario_meta["last_replayed_batch_id"] == expected_first_batch


def test_task5_prioritize_incremental_batch_replays_one_batch_at_a_time() -> None:
    env = PipelineDoctorEnvironment()
    env.reset(task_id=5, split="eval", seed=1)

    initial_backlog = env.state.backlog_rows
    initial_batches = env.state.pending_batches
    expected_first_batch = sorted(
        env._scenario_meta["pending_orders"]["batch_id"].astype(str).unique().tolist()
    )[0]

    env.step(PipelineDoctorAction(action_id=16))
    env.step(PipelineDoctorAction(action_id=16))
    replay = env.step(PipelineDoctorAction(action_id=18))

    assert replay.done is False
    assert env.state.backlog_rows < initial_backlog
    assert env.state.backlog_rows > 0
    assert env.state.pending_batches == initial_batches - 1
    assert env._scenario_meta["last_replayed_batch_id"] == expected_first_batch
