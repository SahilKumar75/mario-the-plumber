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
from openenv.core.env_server.types import EnvironmentMetadata

from benchmark.actions.dispatch import apply_action
from benchmark.actions.orchestration import (
    commit_changes,
    task3_commit_ready,
    task4_commit_ready,
    task5_commit_ready,
)
from benchmark.actions.validation import table_has_structural_issues
from benchmark.catalog import MAX_STEPS
from benchmark.diagnostics import refresh_errors
from benchmark.env_reporting import build_observation
from benchmark.evaluation import score, store_episode_summary
from benchmark.observation_support import workload_pressure
from benchmark.progress import update_task_progress_state
from benchmark.task_ids import parse_task_id
from models import PipelineDoctorAction, PipelineDoctorObservation, PipelineDoctorState
from server.data_generator import generate_scenario
from server.runtime import MIN_VALIDATOR_EXPOSED_SCORE, initialize_episode, resolve_step

EPISODE_SUMMARIES: dict[str, dict[str, object]] = {}


class PipelineDoctorEnvironment(
    Environment[PipelineDoctorAction, PipelineDoctorObservation, PipelineDoctorState]
):
    """Environment where agents diagnose and repair broken synthetic pipeline tables."""

    SUPPORTS_CONCURRENT_SESSIONS = False

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
            current_score=MIN_VALIDATOR_EXPOSED_SCORE,
            initial_score=MIN_VALIDATOR_EXPOSED_SCORE,
            best_score=MIN_VALIDATOR_EXPOSED_SCORE,
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
            last_action_id=None,
            repeated_action_streak=0,
            repeated_action_tripwire=False,
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
        task_id: int | str = 1,
        split: str = "train",
        **_: object,
    ) -> PipelineDoctorObservation:
        """Reset the environment to a fresh synthetic scenario."""

        task_id = parse_task_id(task_id)
        scenario = generate_scenario(task_id=task_id, seed=seed, split=split)
        initialize_episode(self, scenario, task_id=task_id, seed=seed, episode_id=episode_id)
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

        self._state.step_count += 1
        if self._state.last_action_id == action.action_id:
            self._state.repeated_action_streak += 1
        else:
            self._state.last_action_id = action.action_id
            self._state.repeated_action_streak = 1
        self._state.repeated_action_tripwire = self._state.repeated_action_streak > 3
        self._ensure_active_table()

        try:
            action_result = self._apply_action(action)
        except ValueError as exc:
            action_valid = False
            action_result = f"invalid: {exc}"

        if action.action_id == 15 and action_valid:
            self._commit_changes()

        resolution = resolve_step(
            self,
            action_id=action.action_id,
            score_before=score_before,
            action_valid=action_valid,
        )
        self._last_reward_breakdown = resolution.reward_breakdown
        observation = self._build_observation(
            reward=resolution.reward,
            done=resolution.done,
            action_result=action_result,
            metadata={
                "action_id": action.action_id,
                "action_name": resolution.action_name,
                "truncated": resolution.truncated,
                "done_reason": resolution.done_reason,
            },
        )
        self._store_episode_summary()
        return observation

    @property
    def state(self) -> PipelineDoctorState:
        """Return current episode metadata."""

        return self._state

    def _apply_action(self, action: PipelineDoctorAction) -> str:
        return apply_action(self, action)

    def _build_observation(
        self,
        *,
        reward: float,
        done: bool,
        action_result: str = "",
        metadata: dict[str, object] | None = None,
    ) -> PipelineDoctorObservation:
        return build_observation(
            self,
            reward=reward,
            done=done,
            action_result=action_result,
            metadata=metadata,
        )

    def _commit_changes(self) -> None:
        commit_changes(self)

    def _commit_ready(self) -> bool:
        if self._task_id == 3:
            return task3_commit_ready(self)
        if self._task_id == 4:
            return task4_commit_ready(self)
        if self._task_id == 5:
            return task5_commit_ready(self)
        return True

    def _current_frame(self) -> pd.DataFrame:
        self._ensure_active_table()
        return self._tables[self._state.active_table]

    def _current_table(self) -> pd.DataFrame:
        self._ensure_active_table()
        return self._tables[self._state.active_table]

    def _ensure_active_table(self) -> None:
        table_name = self._state.active_table
        if table_name in self._tables:
            active_table = table_name
        elif self._tables:
            active_table = next(iter(self._tables))
            self._state.active_table = active_table
        else:
            active_table = table_name
            self._tables[active_table] = pd.DataFrame()

        self._expected_types.setdefault(active_table, {})
        self._ground_truth.setdefault(active_table, pd.DataFrame())

    def _refresh_errors(self) -> None:
        refresh_errors(self)

    def _score(self) -> float:
        return score(self)

    def _store_episode_summary(self) -> None:
        store_episode_summary(self)

    def get_metadata(self) -> EnvironmentMetadata:
        """Return concrete environment metadata instead of the generic base placeholder."""

        return EnvironmentMetadata(
            name="mario_the_plumber",
            description="Benchmark environment for repairing broken ETL pipelines across five graded tasks.",
            version="2.1",
            author="Team SST",
        )

    def _table_has_structural_issues(self, table_name: str) -> bool:
        return table_has_structural_issues(self, table_name)

    def _task3_commit_ready(self) -> bool:
        return task3_commit_ready(self)

    def _task4_commit_ready(self) -> bool:
        return task4_commit_ready(self)

    def _task5_commit_ready(self) -> bool:
        return task5_commit_ready(self)

    def _update_task_progress_state(self) -> None:
        update_task_progress_state(self)

    def _workload_pressure(self) -> float:
        return workload_pressure(self)
