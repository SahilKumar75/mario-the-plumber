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

    action_id: int = Field(..., ge=0, le=15, description="Discrete action id.")
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

    missing_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    duplicate_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    type_violations: int = Field(default=0, ge=0)
    outlier_count: int = Field(default=0, ge=0)
    schema_report: dict[str, dict[str, Any]] = Field(default_factory=dict)
    recent_errors: list[str] = Field(default_factory=list)
    current_score: float = Field(default=0.0, ge=0.0, le=1.0)
    steps_taken: int = Field(default=0, ge=0)
    stage: str = Field(default="single")
    available_actions: list[int] = Field(default_factory=lambda: list(range(16)))
    action_result: str = Field(default="")


class MarioThePlumberState(State):
    """Internal episode metadata surfaced via the OpenEnv state endpoint."""

    task_id: int = Field(default=1, ge=1, le=3)
    seed: int | None = None
    max_steps: int = Field(default=10, ge=1)
    current_score: float = Field(default=0.0, ge=0.0, le=1.0)
    initial_score: float = Field(default=0.0, ge=0.0, le=1.0)
    best_score: float = Field(default=0.0, ge=0.0, le=1.0)
    done: bool = False
    success: bool | None = None
    active_table: str = Field(default="single")
    started_at: str = Field(default="")


PipelineDoctorAction = MarioThePlumberAction
PipelineDoctorObservation = MarioThePlumberObservation
PipelineDoctorState = MarioThePlumberState
