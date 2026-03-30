# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Core OpenEnv environment for PipelineDoctor."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

import pandas as pd
from openenv.core.env_server.interfaces import Environment

try:
    from ..models import PipelineDoctorAction, PipelineDoctorObservation, PipelineDoctorState
    from .data_generator import MAX_STEPS, TASK_THRESHOLDS, generate_scenario
    from .grader import calculation_mismatch_count, compute_reward, score_single_table, score_task3
except ImportError:
    from models import PipelineDoctorAction, PipelineDoctorObservation, PipelineDoctorState
    from server.data_generator import MAX_STEPS, TASK_THRESHOLDS, generate_scenario
    from server.grader import calculation_mismatch_count, compute_reward, score_single_table, score_task3

ACTION_NAMES = {
    0: "inspect_schema",
    1: "view_error_log",
    2: "sample_data",
    3: "fill_mean",
    4: "fill_median",
    5: "fill_forward",
    6: "drop_nulls",
    7: "cast_to_int",
    8: "cast_to_float",
    9: "cast_to_string",
    10: "remove_duplicates",
    11: "drop_outliers",
    12: "rename_column",
    13: "reorder_columns",
    14: "validate_schema",
    15: "commit_changes",
}
PARAMETER_ACTIONS = {3, 4, 5, 6, 7, 8, 9, 11, 12}
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
        self._task_id = 1
        self._seed: int | None = None
        self._recent_errors: list[str] = []
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
            started_at=datetime.now(UTC).isoformat(),
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: int = 1,
        **_: object,
    ) -> PipelineDoctorObservation:
        """Reset the environment to a fresh synthetic scenario."""

        scenario = generate_scenario(task_id=task_id, seed=seed)
        self._tables = {
            name: frame.copy(deep=True) for name, frame in scenario.broken_tables.items()
        }
        self._ground_truth = {
            name: frame.copy(deep=True) for name, frame in scenario.ground_truth_tables.items()
        }
        self._expected_types = dict(scenario.expected_types)
        self._task_id = task_id
        self._seed = seed
        self._recent_errors = []
        current_score = self._score()

        self._state = PipelineDoctorState(
            episode_id=episode_id or str(uuid4()),
            task_id=task_id,
            seed=seed,
            step_count=0,
            max_steps=MAX_STEPS[task_id],
            current_score=current_score,
            initial_score=current_score,
            best_score=current_score,
            done=False,
            success=None,
            active_table=scenario.active_table,
            started_at=datetime.now(UTC).isoformat(),
        )
        self._refresh_errors()
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
        elif score_after < 0.10:
            done = True
        elif self._state.step_count >= self._state.max_steps:
            done = True

        reward = compute_reward(
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
        observation = self._build_observation(
            reward=reward,
            done=done,
            action_result=action_result,
            metadata={"action_id": action.action_id, "action_name": action_name},
        )
        self._store_episode_summary()
        return observation

    @property
    def state(self) -> PipelineDoctorState:
        """Return current episode metadata."""

        return self._state

    def _apply_action(self, action: PipelineDoctorAction) -> str:
        if action.action_id not in ACTION_NAMES:
            raise ValueError("unknown action_id")

        if action.action_id == 0:
            return self._handle_inspect_schema(action)
        if action.action_id == 1:
            return "\n".join(self._recent_errors) if self._recent_errors else "No errors detected."
        if action.action_id == 2:
            rows = self._current_table().head(5).to_dict(orient="records")
            return json.dumps(rows, default=str)
        if action.action_id in PARAMETER_ACTIONS and not action.target_column:
            raise ValueError("missing required parameter target_column")
        if action.action_id == 12 and not action.new_name:
            raise ValueError("missing required parameter new_name")
        if action.action_id == 13 and not action.column_order:
            raise ValueError("missing required parameter column_order")

        if action.action_id == 3:
            self._fill_with_statistic(action.target_column, "mean")
        elif action.action_id == 4:
            self._fill_with_statistic(action.target_column, "median")
        elif action.action_id == 5:
            self._current_frame()[action.target_column] = (
                self._current_frame()[action.target_column].ffill().bfill()
            )
        elif action.action_id == 6:
            current = self._current_frame()
            self._tables[self._state.active_table] = current[current[action.target_column].notna()].reset_index(drop=True)
        elif action.action_id == 7:
            self._cast_column(action.target_column, "int64")
        elif action.action_id == 8:
            self._cast_column(action.target_column, "float64")
        elif action.action_id == 9:
            self._current_frame()[action.target_column] = self._current_frame()[
                action.target_column
            ].astype(str)
        elif action.action_id == 10:
            self._tables[self._state.active_table] = self._current_frame().drop_duplicates().reset_index(drop=True)
        elif action.action_id == 11:
            self._drop_outliers(action.target_column)
        elif action.action_id == 12:
            if action.target_column not in self._current_frame().columns:
                raise ValueError("target_column not found")
            self._tables[self._state.active_table] = self._current_frame().rename(
                columns={action.target_column: action.new_name}
            )
        elif action.action_id == 13:
            if set(action.column_order) != set(self._current_frame().columns):
                raise ValueError("column_order must match current table columns exactly")
            self._tables[self._state.active_table] = self._current_frame()[action.column_order].copy()
        elif action.action_id == 14:
            return "\n".join(self._recent_errors) if self._recent_errors else "Schema validation passed."
        elif action.action_id == 15:
            return "Changes committed."

        return ""

    def _handle_inspect_schema(self, action: PipelineDoctorAction) -> str:
        if action.target_column and action.target_column in self._tables:
            self._state.active_table = action.target_column
        current = self._current_frame()
        expected = self._expected_types[self._state.active_table]
        lines = [f"table={self._state.active_table}"]
        for column in current.columns:
            lines.append(
                f"{column}: actual={current[column].dtype}, expected={expected.get(column, 'unknown')}"
            )
        return "\n".join(lines)

    def _fill_with_statistic(self, column: str | None, statistic: str) -> None:
        if column not in self._current_frame().columns:
            raise ValueError("target_column not found")
        numeric = pd.to_numeric(self._current_frame()[column], errors="coerce")
        if statistic == "mean":
            value = numeric.mean()
        else:
            value = numeric.median()
        self._current_frame()[column] = numeric.fillna(value)

    def _cast_column(self, column: str | None, dtype: str) -> None:
        if column not in self._current_frame().columns:
            raise ValueError("target_column not found")
        numeric = pd.to_numeric(self._current_frame()[column], errors="coerce")
        if dtype == "int64":
            if numeric.isna().any():
                raise ValueError("cannot cast to int while nulls remain")
            self._current_frame()[column] = numeric.astype("int64")
        else:
            self._current_frame()[column] = numeric.astype("float64")

    def _drop_outliers(self, column: str | None) -> None:
        if column not in self._current_frame().columns:
            raise ValueError("target_column not found")
        numeric = pd.to_numeric(self._current_frame()[column], errors="coerce")
        std = float(numeric.std(skipna=True))
        if pd.isna(std) or std == 0:
            return
        mean = float(numeric.mean(skipna=True))
        mask = (numeric - mean).abs() <= 3 * std
        mask = mask.fillna(False)
        self._tables[self._state.active_table] = self._current_frame()[mask].reset_index(drop=True)

    def _commit_changes(self) -> None:
        if self._task_id != 3:
            return
        orders = self._tables["orders"].copy()
        products = self._tables["products"][["product_id", "unit_price"]].copy()
        merged = orders.merge(products, on="product_id", how="left", suffixes=("", "_product"))
        quantity = pd.to_numeric(merged["quantity"], errors="coerce")
        unit_price = pd.to_numeric(merged["unit_price"], errors="coerce")
        merged["total_price"] = (quantity * unit_price).round(2)
        self._tables["orders"] = merged[orders.columns].copy()

    def _build_observation(
        self,
        *,
        reward: float,
        done: bool,
        action_result: str = "",
        metadata: dict[str, object] | None = None,
    ) -> PipelineDoctorObservation:
        current = self._current_frame()
        total_cells = max(len(current) * max(len(current.columns), 1), 1)
        missing_rate = float(current.isnull().sum().sum() / total_cells)
        duplicate_rate = float(current.duplicated().sum() / max(len(current), 1))

        schema_report = self._schema_report()
        observation = PipelineDoctorObservation(
            missing_rate=round(missing_rate, 4),
            duplicate_rate=round(duplicate_rate, 4),
            type_violations=len(schema_report),
            outlier_count=self._outlier_count(),
            schema_report=schema_report,
            recent_errors=self._recent_errors[:5],
            current_score=self._state.current_score,
            steps_taken=self._state.step_count,
            stage=self._state.active_table,
            available_actions=list(range(16)),
            action_result=action_result,
            reward=reward,
            done=done,
            metadata=metadata or {},
        )
        return observation

    def _schema_report(self) -> dict[str, dict[str, str]]:
        current = self._current_frame()
        expected = self._expected_types[self._state.active_table]
        report: dict[str, dict[str, str]] = {}
        for column in current.columns:
            actual = str(current[column].dtype)
            desired = expected.get(column)
            if desired and actual != desired:
                report[column] = {"expected": desired, "actual": actual}
        for column in expected:
            if column not in current.columns:
                report[column] = {"expected": expected[column], "actual": "missing"}
        return report

    def _outlier_details(self) -> dict[str, int]:
        details: dict[str, int] = {}
        current = self._current_frame()
        for column in current.columns:
            numeric = pd.to_numeric(current[column], errors="coerce")
            if numeric.isna().all():
                continue
            std = float(numeric.std(skipna=True))
            if pd.isna(std) or std == 0:
                continue
            mean = float(numeric.mean(skipna=True))
            outlier_count = int(((numeric - mean).abs() > 3 * std).fillna(False).sum())
            if outlier_count > 0:
                details[column] = outlier_count
        return details

    def _outlier_count(self) -> int:
        count = 0
        for column_count in self._outlier_details().values():
            count += column_count
        return count

    def _refresh_errors(self) -> None:
        current = self._current_frame()
        errors: list[str] = []
        null_counts = current.isnull().sum()
        for column, count in null_counts.items():
            if int(count) > 0:
                errors.append(f"{column}: {int(count)} null values")
        duplicate_count = int(current.duplicated().sum())
        if duplicate_count > 0:
            errors.append(f"{duplicate_count} duplicate rows detected")
        for column, count in self._outlier_details().items():
            errors.append(f"{column}: {count} outlier values")
        for column, info in self._schema_report().items():
            errors.append(
                f"{column}: expected {info['expected']}, found {info['actual']}"
            )
        if self._task_id == 3 and self._state.active_table == "orders":
            mismatch_count = calculation_mismatch_count(
                self._tables["orders"], self._tables["products"]
            )
            if mismatch_count > 0:
                errors.append(f"total_price: {mismatch_count} rows have calculation mismatch")
        self._recent_errors = errors[:5]

    def _score(self) -> float:
        if self._task_id == 3:
            score, _ = score_task3(self._tables, self._ground_truth, self._expected_types)
            return score
        score, _ = score_single_table(
            self._tables["single"],
            self._ground_truth["single"],
            self._expected_types["single"],
        )
        return score

    def _current_table(self) -> pd.DataFrame:
        return self._tables[self._state.active_table]

    def _current_frame(self) -> pd.DataFrame:
        return self._tables[self._state.active_table]

    def _store_episode_summary(self) -> None:
        threshold = TASK_THRESHOLDS[self._task_id]
        success = bool(self._state.done and self._state.current_score >= threshold)
        summary = {
            "task_id": self._task_id,
            "episode_id": self._state.episode_id,
            "score": round(self._state.current_score, 4),
            "breakdown": self._breakdown_payload(),
            "success": success,
            "steps_taken": self._state.step_count,
        }
        EPISODE_SUMMARIES[self._state.episode_id or str(uuid4())] = summary

    def _breakdown_payload(self) -> dict[str, object]:
        if self._task_id == 3:
            _, breakdown = score_task3(self._tables, self._ground_truth, self._expected_types)
            return breakdown
        _, breakdown = score_single_table(
            self._tables["single"],
            self._ground_truth["single"],
            self._expected_types["single"],
        )
        return breakdown
