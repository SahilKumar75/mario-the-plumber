# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client wrapper for the Mario the Plumber environment."""

from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import MarioThePlumberAction, MarioThePlumberObservation, MarioThePlumberState


class MarioThePlumberEnv(
    EnvClient[MarioThePlumberAction, MarioThePlumberObservation, MarioThePlumberState]
):
    """Typed OpenEnv client for interacting with a running Mario the Plumber server."""

    def _step_payload(self, action: MarioThePlumberAction) -> dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(
        self, payload: dict[str, Any]
    ) -> StepResult[MarioThePlumberObservation]:
        obs_data = payload.get("observation", {})
        observation = MarioThePlumberObservation(
            missing_rate=obs_data.get("missing_rate", 0.0),
            duplicate_rate=obs_data.get("duplicate_rate", 0.0),
            type_violations=obs_data.get("type_violations", 0),
            outlier_count=obs_data.get("outlier_count", 0),
            schema_report=obs_data.get("schema_report", {}),
            recent_errors=obs_data.get("recent_errors", []),
            current_score=obs_data.get("current_score", 0.0),
            steps_taken=obs_data.get("steps_taken", 0),
            stage=obs_data.get("stage", "single"),
            available_actions=obs_data.get("available_actions", list(range(16))),
            action_result=obs_data.get("action_result", ""),
            reward=payload.get("reward", obs_data.get("reward")),
            done=payload.get("done", obs_data.get("done", False)),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> MarioThePlumberState:
        return MarioThePlumberState(**payload)


PipelineDoctorEnv = MarioThePlumberEnv
