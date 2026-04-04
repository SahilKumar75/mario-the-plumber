# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Core OpenEnv environment for PipelineDoctor."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pandas as pd
from openenv.core.env_server.interfaces import Environment

try:
    from ..benchmark.action_metadata import ACTION_NAMES
    from ..benchmark.catalog import (
        MAX_STEPS,
        TASK_THRESHOLDS,
    )
    from ..benchmark.grading import (
        compute_reward,
        compute_reward_breakdown,
    )
    from ..models import PipelineDoctorAction, PipelineDoctorObservation, PipelineDoctorState
    from .environment_support import EnvironmentSupportMixin
    from .data_generator import generate_scenario
except ImportError:
    from benchmark.action_metadata import ACTION_NAMES
    from benchmark.catalog import (
        MAX_STEPS,
        TASK_THRESHOLDS,
    )
    from benchmark.grading import (
        compute_reward,
        compute_reward_breakdown,
    )
    from models import PipelineDoctorAction, PipelineDoctorObservation, PipelineDoctorState
    from server.environment_support import EnvironmentSupportMixin
    from server.data_generator import generate_scenario

EPISODE_SUMMARIES: dict[str, dict[str, object]] = {}


class PipelineDoctorEnvironment(
    EnvironmentSupportMixin,
    Environment[PipelineDoctorAction, PipelineDoctorObservation, PipelineDoctorState]
):
    """Environment where agents diagnose and repair broken synthetic pipeline tables."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._tables: dict[str, pd.DataFrame] = {}
        self._ground_truth: dict[str, pd.DataFrame] = {}
        self._expected_types: dict[str, dict[str, str]] = {}
        self._scenario_meta: dict[str, object] = {}
        self._task_id = 1
        self._seed: int | None = None
        self._split = "train"
        self._recent_errors: list[str] = []
        self._last_reward_breakdown: dict[str, float] = {}
        self._episode_summaries = EPISODE_SUMMARIES
        self._state = PipelineDoctorState(
            episode_id=str(uuid4()),
            task_id=1,
            seed=None,
            max_steps=MAX_STEPS[1],
            current_score=0.0,
            initial_score=0.0,
            best_score=0.0,
            done=False,
            success=None,
            active_table="single",
            scenario_split="train",
            backlog_rows=0,
            queue_backlog_age_minutes=0,
            freshness_lag_minutes=0,
            sla_severity="none",
            resource_level=1,
            required_resource_level=1,
            pending_batches=0,
            time_budget_remaining=MAX_STEPS[1],
            truncated=False,
            done_reason="",
            scenario_profile="baseline",
            started_at=datetime.now(UTC).isoformat(),
            active_subgoal="",
            reward_machine_state="",
            heldout_profile_family=False,
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: int = 1,
        split: str = "train",
        **_: object,
    ) -> PipelineDoctorObservation:
        """Reset the environment to a fresh synthetic scenario."""

        scenario = generate_scenario(task_id=task_id, seed=seed, split=split)
        self._tables = {
            name: frame.copy(deep=True) for name, frame in scenario.broken_tables.items()
        }
        self._ground_truth = {
            name: frame.copy(deep=True) for name, frame in scenario.ground_truth_tables.items()
        }
        self._expected_types = dict(scenario.expected_types)
        self._scenario_meta = {
            key: value.copy(deep=True) if isinstance(value, pd.DataFrame) else value
            for key, value in scenario.metadata.items()
        }
        self._task_id = task_id
        self._seed = seed
        self._split = scenario.split
        self._recent_errors = []

        self._state = PipelineDoctorState(
            episode_id=episode_id or str(uuid4()),
            task_id=task_id,
            seed=seed,
            step_count=0,
            max_steps=MAX_STEPS[task_id],
            current_score=0.0,
            initial_score=0.0,
            best_score=0.0,
            done=False,
            success=None,
            active_table=scenario.active_table,
            scenario_split=scenario.split,
            backlog_rows=int(self._scenario_meta.get("backlog_rows", 0)),
            queue_backlog_age_minutes=int(self._scenario_meta.get("queue_backlog_age_minutes", 0)),
            freshness_lag_minutes=int(self._scenario_meta.get("freshness_lag_minutes", 0)),
            sla_severity="none",
            resource_level=int(self._scenario_meta.get("resource_level", 1)),
            required_resource_level=int(self._scenario_meta.get("required_resource_level", 1)),
            pending_batches=int(self._scenario_meta.get("pending_batches", 0)),
            time_budget_remaining=MAX_STEPS[task_id],
            truncated=False,
            done_reason="",
            scenario_profile=str(self._scenario_meta.get("scenario_profile", "baseline")),
            started_at=datetime.now(UTC).isoformat(),
            active_subgoal="",
            reward_machine_state="",
            heldout_profile_family=bool(self._scenario_meta.get("heldout_profile_family", False)),
        )
        self._refresh_errors()
        self._update_task_progress_state()
        current_score = self._score()
        self._state.current_score = current_score
        self._state.initial_score = current_score
        self._state.best_score = current_score
        self._store_episode_summary()
        return self._build_observation(reward=0.0, done=False)

    def step(
        self,
        action: PipelineDoctorAction,
        timeout_s: float | None = None,
        **_: object,
    ) -> PipelineDoctorObservation:
        """Apply one discrete repair or inspection action."""

        del timeout_s
        score_before = self._state.current_score
        action_valid = True
        action_result = ""
        done = False
        success = False
        truncated = False
        done_reason = ""

        self._state.step_count += 1
        action_name = ACTION_NAMES.get(action.action_id, "unknown")

        try:
            action_result = self._apply_action(action)
        except ValueError as exc:
            action_valid = False
            action_result = f"invalid: {exc}"

        if action.action_id == 15 and action_valid:
            self._commit_changes()

        self._refresh_errors()
        commit_ready = self._commit_ready()
        score_after = self._score()

        threshold = TASK_THRESHOLDS[self._task_id]
        if action.action_id == 15:
            done = True
            success = commit_ready and score_after >= threshold and action_valid
            done_reason = "commit_success" if success else "commit_failure"
        elif score_after < 0.10:
            done = True
            done_reason = "quality_collapse"
        elif self._state.step_count >= self._state.max_steps:
            done = True
            truncated = True
            done_reason = "step_budget_exhausted"

        reward = compute_reward(
            score_before,
            score_after,
            action_valid=action_valid,
            done=done,
            success=success,
        )
        self._last_reward_breakdown = compute_reward_breakdown(
            score_before,
            score_after,
            action_valid=action_valid,
            done=done,
            success=success,
        )

        self._state.current_score = score_after
        self._state.best_score = max(self._state.best_score, score_after)
        self._state.done = done
        self._state.success = success if done else None
        self._state.truncated = truncated
        self._state.done_reason = done_reason
        self._state.time_budget_remaining = max(
            0, self._state.max_steps - self._state.step_count
        )
        self._update_task_progress_state()
        observation = self._build_observation(
            reward=reward,
            done=done,
            action_result=action_result,
            metadata={
                "action_id": action.action_id,
                "action_name": action_name,
                "truncated": truncated,
                "done_reason": done_reason,
            },
        )
        self._store_episode_summary()
        return observation

    @property
    def state(self) -> PipelineDoctorState:
        """Return current episode metadata."""

        return self._state
