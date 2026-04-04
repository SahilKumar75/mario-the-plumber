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
