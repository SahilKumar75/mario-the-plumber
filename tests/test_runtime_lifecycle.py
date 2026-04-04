from __future__ import annotations

from models import PipelineDoctorAction
from server.pipeline_doctor_environment import PipelineDoctorEnvironment


def test_reset_restores_budget_and_deterministic_profile() -> None:
    env = PipelineDoctorEnvironment()
    first = env.reset(task_id=4, split="eval", seed=9)
    env.step(PipelineDoctorAction(action_id=0))
    second = env.reset(task_id=4, split="eval", seed=9)

    assert first.scenario_profile == second.scenario_profile
    assert first.current_score == second.current_score
    assert second.steps_taken == 0
    assert env.state.step_count == 0
    assert env.state.time_budget_remaining == env.state.max_steps


def test_valid_step_consumes_budget_without_ending_episode() -> None:
    env = PipelineDoctorEnvironment()
    env.reset(task_id=1, split="train", seed=1)

    observation = env.step(PipelineDoctorAction(action_id=0))

    assert observation.done is False
    assert observation.steps_taken == 1
    assert env.state.step_count == 1
    assert env.state.time_budget_remaining == env.state.max_steps - 1
    assert observation.action_result
    assert observation.reward_breakdown["step_cost"] == -0.001
