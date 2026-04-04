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
    from ..benchmark.env_actions import (
        apply_action,
        cast_column,
        commit_changes,
        deduplicate_current_table,
        drop_outliers,
        fill_with_statistic,
        handle_inspect_schema,
        normalize_date_string,
        normalize_numeric_value,
        normalize_string_value,
        numeric_series,
        prioritize_incremental_batch,
        refresh_hourly_rollup,
        refresh_downstream_summary,
        scale_resources,
        table_has_structural_issues,
        task3_commit_ready,
        task4_commit_ready,
        task5_commit_ready,
    )
    from ..benchmark.env_reporting import (
        breakdown_payload,
        build_observation,
        format_issue_count,
        format_issue_details,
        objective_breakdown,
        outlier_count,
        outlier_details,
        refresh_errors,
        schema_report,
        schema_report_for_table,
        score,
        store_episode_summary,
        subgoal_progress_map,
        task_progress_bundle,
        update_task_progress_state,
    )
    from ..benchmark.grading import (
        compute_reward,
        compute_reward_breakdown,
    )
    from ..benchmark.observation_support import (
        column_alias_hints,
        dependency_alerts,
        format_issue_details_for_frame,
        missing_expected_columns,
        orchestration_alerts,
        outlier_details_for_frame,
        table_health,
        workload_pressure,
    )
    from ..models import PipelineDoctorAction, PipelineDoctorObservation, PipelineDoctorState
    from .data_generator import generate_scenario
except ImportError:
    from benchmark.action_metadata import ACTION_NAMES
    from benchmark.catalog import (
        MAX_STEPS,
        TASK_THRESHOLDS,
    )
    from benchmark.env_actions import (
        apply_action,
        cast_column,
        commit_changes,
        deduplicate_current_table,
        drop_outliers,
        fill_with_statistic,
        handle_inspect_schema,
        normalize_date_string,
        normalize_numeric_value,
        normalize_string_value,
        numeric_series,
        prioritize_incremental_batch,
        refresh_hourly_rollup,
        refresh_downstream_summary,
        scale_resources,
        table_has_structural_issues,
        task3_commit_ready,
        task4_commit_ready,
        task5_commit_ready,
    )
    from benchmark.env_reporting import (
        breakdown_payload,
        build_observation,
        format_issue_count,
        format_issue_details,
        objective_breakdown,
        outlier_count,
        outlier_details,
        refresh_errors,
        schema_report,
        schema_report_for_table,
        score,
        store_episode_summary,
        subgoal_progress_map,
        task_progress_bundle,
        update_task_progress_state,
    )
    from benchmark.grading import (
        compute_reward,
        compute_reward_breakdown,
    )
    from benchmark.observation_support import (
        column_alias_hints,
        dependency_alerts,
        format_issue_details_for_frame,
        missing_expected_columns,
        orchestration_alerts,
        outlier_details_for_frame,
        table_health,
        workload_pressure,
    )
    from models import PipelineDoctorAction, PipelineDoctorObservation, PipelineDoctorState
    from server.data_generator import generate_scenario

EPISODE_SUMMARIES: dict[str, dict[str, object]] = {}


class PipelineDoctorEnvironment(
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
        current_score = self._score()
        self._state.current_score = current_score
        self._state.initial_score = current_score
        self._state.best_score = current_score
        self._refresh_errors()
        self._update_task_progress_state()
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
        score_after = self._score()

        threshold = TASK_THRESHOLDS[self._task_id]
        if action.action_id == 15:
            done = True
            success = score_after >= threshold and action_valid
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

    def _apply_action(self, action: PipelineDoctorAction) -> str:
        return apply_action(self, action)

    def _handle_inspect_schema(self, action: PipelineDoctorAction) -> str:
        return handle_inspect_schema(self, action)

    def _fill_with_statistic(self, column: str | None, statistic: str) -> None:
        fill_with_statistic(self, column, statistic)

    def _cast_column(self, column: str | None, dtype: str) -> None:
        cast_column(self, column, dtype)

    def _numeric_series(self, column: str) -> pd.Series:
        return numeric_series(self, column)

    def _normalize_numeric_value(self, value: object) -> object:
        return normalize_numeric_value(value)

    def _normalize_string_value(self, value: object, column: str | None) -> object:
        return normalize_string_value(self, value, column)

    def _normalize_date_string(self, text: str, *, preserve_time: bool = False) -> str | None:
        return normalize_date_string(text, preserve_time=preserve_time)

    def _is_datetime_like_column(self, column_name: str) -> bool:
        return "date" in column_name or column_name.endswith("_ts") or column_name.endswith("_time")

    def _looks_timestamp_string(self, text: str) -> bool:
        return ":" in text

    def _drop_outliers(self, column: str | None) -> None:
        drop_outliers(self, column)

    def _commit_changes(self) -> None:
        commit_changes(self)

    def _task3_commit_ready(self) -> bool:
        return task3_commit_ready(self)

    def _task4_commit_ready(self) -> bool:
        return task4_commit_ready(self)

    def _task5_commit_ready(self) -> bool:
        return task5_commit_ready(self)

    def _scale_resources(self, *, up: bool) -> str:
        return scale_resources(self, up=up)

    def _prioritize_incremental_batch(self) -> str:
        return prioritize_incremental_batch(self)

    def _refresh_downstream_summary(self) -> str:
        return refresh_downstream_summary(self)

    def _refresh_hourly_rollup(self) -> str:
        return refresh_hourly_rollup(self)

    def _table_has_structural_issues(self, table_name: str) -> bool:
        return table_has_structural_issues(self, table_name)

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

    def _schema_report(self) -> dict[str, dict[str, str]]:
        return schema_report(self)

    def _deduplicate_current_table(self) -> pd.DataFrame:
        return deduplicate_current_table(self)

    def _schema_report_for_table(self, table_name: str) -> dict[str, dict[str, str]]:
        return schema_report_for_table(self, table_name)

    def _missing_expected_columns(self, table_name: str) -> list[str]:
        return missing_expected_columns(self, table_name)

    def _column_alias_hints(self) -> dict[str, str]:
        return column_alias_hints(self)

    def _outlier_details(self) -> dict[str, int]:
        return outlier_details(self)

    def _outlier_details_for_frame(self, current: pd.DataFrame) -> dict[str, int]:
        return outlier_details_for_frame(self, current)

    def _outlier_count(self) -> int:
        return outlier_count(self)

    def _format_issue_details(self) -> dict[str, int]:
        return format_issue_details(self)

    def _format_issue_details_for_frame(self, current: pd.DataFrame) -> dict[str, int]:
        return format_issue_details_for_frame(self, current)

    def _format_issue_count(self) -> int:
        return format_issue_count(self)

    def _dependency_alerts(self) -> list[str]:
        return dependency_alerts(self)

    def _table_health(self) -> dict[str, float]:
        return table_health(self)

    def _refresh_errors(self) -> None:
        refresh_errors(self)

    def _score(self) -> float:
        return score(self)

    def _current_table(self) -> pd.DataFrame:
        return self._tables[self._state.active_table]

    def _current_frame(self) -> pd.DataFrame:
        return self._tables[self._state.active_table]

    def _store_episode_summary(self) -> None:
        self.EPISODE_SUMMARIES = EPISODE_SUMMARIES
        store_episode_summary(self)

    def _breakdown_payload(self) -> dict[str, object]:
        return breakdown_payload(self)

    def _workload_pressure(self) -> float:
        return workload_pressure(self)

    def _orchestration_alerts(self) -> list[str]:
        return orchestration_alerts(self)

    def _commit_ready(self) -> bool:
        if self._task_id == 3:
            return self._task3_commit_ready()
        if self._task_id == 4:
            return self._task4_commit_ready()
        if self._task_id == 5:
            return self._task5_commit_ready()
        return True

    def _objective_breakdown(self) -> dict[str, float]:
        return objective_breakdown(self)

    def _task_progress_bundle(self) -> tuple[dict[str, bool], list[str], str, str]:
        return task_progress_bundle(self)

    def _update_task_progress_state(self) -> None:
        update_task_progress_state(self)

    def _subgoal_progress(self) -> dict[str, bool]:
        return subgoal_progress_map(self)
