# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed models for the Mario the Plumber OpenEnv environment."""

from typing import Any

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class MarioThePlumberAction(Action):
    """Discrete pipeline repair action with optional parameters."""

    action_id: int = Field(..., ge=0, le=19, description="Discrete action id.")
    target_column: str | None = Field(
        default=None,
        description=(
            "Required for actions 3-9, 11, and 12. Optional for action 0 in task 3 "
            "to switch the active table."
        ),
    )
    new_name: str | None = Field(
        default=None, description="Required for action 12 (rename_column)."
    )
    column_order: list[str] | None = Field(
        default=None, description="Required for action 13 (reorder_columns)."
    )


class MarioThePlumberObservation(Observation):
    """Quality-signal observation returned after reset and step."""

    incident_type: str = Field(default="")
    incident_summary: str = Field(default="")
    diagnosis_signals: list[str] = Field(default_factory=list)
    recovery_requirements: list[str] = Field(default_factory=list)
    unsafe_commit_conditions: list[str] = Field(default_factory=list)
    threshold_rationale: str = Field(default="")
    missing_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    duplicate_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    type_violations: int = Field(default=0, ge=0)
    outlier_count: int = Field(default=0, ge=0)
    format_issues: int = Field(default=0, ge=0)
    schema_report: dict[str, dict[str, Any]] = Field(default_factory=dict)
    recent_errors: list[str] = Field(default_factory=list)
    current_score: float = Field(default=0.0, ge=0.0, le=1.0)
    steps_taken: int = Field(default=0, ge=0)
    stage: str = Field(default="single")
    available_actions: list[int] = Field(default_factory=lambda: list(range(20)))
    action_result: str = Field(default="")
    table_health: dict[str, float] = Field(default_factory=dict)
    dependency_alerts: list[str] = Field(default_factory=list)
    commit_ready: bool = False
    scenario_split: str = Field(default="train")
    schema_drift_count: int = Field(default=0, ge=0)
    backlog_rows: int = Field(default=0, ge=0)
    queue_backlog_age_minutes: int = Field(default=0, ge=0)
    freshness_lag_minutes: int = Field(default=0, ge=0)
    sla_severity: str = Field(default="none")
    resource_level: int = Field(default=1, ge=1)
    required_resource_level: int = Field(default=1, ge=1)
    workload_pressure: float = Field(default=0.0, ge=0.0, le=1.0)
    pending_batches: int = Field(default=0, ge=0)
    downstream_stale: bool = False
    orchestration_alerts: list[str] = Field(default_factory=list)
    recent_failure_counters: dict[str, int] = Field(default_factory=dict)
    drift_markers: list[str] = Field(default_factory=list)
    dependency_health_summary: dict[str, str] = Field(default_factory=dict)
    observed_columns: list[str] = Field(default_factory=list)
    missing_expected_columns: list[str] = Field(default_factory=list)
    column_alias_hints: dict[str, str] = Field(default_factory=dict)
    scenario_profile: str = Field(default="baseline")
    open_world_patterns: list[str] = Field(default_factory=list)
    time_budget_remaining: int = Field(default=0, ge=0)
    time_budget_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    truncated: bool = False
    done_reason: str = Field(default="")
    synthetic_data_notes: list[str] = Field(default_factory=list)
    reward_breakdown: dict[str, float] = Field(default_factory=dict)
    objective_breakdown: dict[str, float] = Field(default_factory=dict)
    tradeoff_weights: dict[str, float] = Field(default_factory=dict)
    subgoal_progress: dict[str, bool] = Field(default_factory=dict)
    subgoal_order: list[str] = Field(default_factory=list)
    active_subgoal: str = Field(default="")
    reward_machine_state: str = Field(default="")
    adaptation_target: str = Field(default="")
    heldout_profile_family: bool = False


class MarioThePlumberState(State):
    """Internal episode metadata surfaced via the OpenEnv state endpoint."""

    task_id: int = Field(default=1, ge=1, le=5)
    seed: int | None = None
    max_steps: int = Field(default=10, ge=1)
    current_score: float = Field(default=0.0, ge=0.0, le=1.0)
    initial_score: float = Field(default=0.0, ge=0.0, le=1.0)
    best_score: float = Field(default=0.0, ge=0.0, le=1.0)
    done: bool = False
    success: bool | None = None
    active_table: str = Field(default="single")
    scenario_split: str = Field(default="train")
    backlog_rows: int = Field(default=0, ge=0)
    queue_backlog_age_minutes: int = Field(default=0, ge=0)
    freshness_lag_minutes: int = Field(default=0, ge=0)
    sla_severity: str = Field(default="none")
    resource_level: int = Field(default=1, ge=1)
    required_resource_level: int = Field(default=1, ge=1)
    pending_batches: int = Field(default=0, ge=0)
    time_budget_remaining: int = Field(default=0, ge=0)
    truncated: bool = False
    done_reason: str = Field(default="")
    scenario_profile: str = Field(default="baseline")
    started_at: str = Field(default="")
    active_subgoal: str = Field(default="")
    reward_machine_state: str = Field(default="")
    heldout_profile_family: bool = False


PipelineDoctorAction = MarioThePlumberAction
PipelineDoctorObservation = MarioThePlumberObservation
PipelineDoctorState = MarioThePlumberState
