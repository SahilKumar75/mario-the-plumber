# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Core OpenEnv environment for PipelineDoctor."""

from __future__ import annotations

from collections import Counter
import json
from datetime import UTC, datetime
import re
from uuid import uuid4

import pandas as pd
from openenv.core.env_server.interfaces import Environment

try:
    from ..models import PipelineDoctorAction, PipelineDoctorObservation, PipelineDoctorState
    from .data_generator import MAX_STEPS, TASK_THRESHOLDS, generate_scenario
    from .grader import (
        calculation_mismatch_count,
        compute_reward,
        duplicate_row_count,
        score_single_table,
        score_task3,
    )
except ImportError:
    from models import PipelineDoctorAction, PipelineDoctorObservation, PipelineDoctorState
    from server.data_generator import MAX_STEPS, TASK_THRESHOLDS, generate_scenario
    from server.grader import (
        calculation_mismatch_count,
        compute_reward,
        duplicate_row_count,
        score_single_table,
        score_task3,
    )

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
        self._split = "train"
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
            scenario_split="train",
            started_at=datetime.now(UTC).isoformat(),
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
        self._task_id = task_id
        self._seed = seed
        self._split = scenario.split
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
            scenario_split=scenario.split,
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
            ].map(lambda value: self._normalize_string_value(value, action.target_column))
        elif action.action_id == 10:
            self._tables[self._state.active_table] = self._deduplicate_current_table()
        elif action.action_id == 11:
            self._drop_outliers(action.target_column)
        elif action.action_id == 12:
            if action.target_column not in self._current_frame().columns:
                raise ValueError("target_column not found")
            if (
                action.new_name != action.target_column
                and action.new_name in self._current_frame().columns
            ):
                raise ValueError("new_name must not duplicate an existing column")
            self._tables[self._state.active_table] = self._current_frame().rename(
                columns={action.target_column: action.new_name}
            )
        elif action.action_id == 13:
            current_columns = list(self._current_frame().columns)
            if Counter(action.column_order) != Counter(current_columns):
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
        numeric = self._numeric_series(column)
        if statistic == "mean":
            value = numeric.mean()
        else:
            value = numeric.median()
        self._current_frame()[column] = numeric.fillna(value)

    def _cast_column(self, column: str | None, dtype: str) -> None:
        if column not in self._current_frame().columns:
            raise ValueError("target_column not found")
        numeric = self._numeric_series(column)
        if dtype == "int64":
            if numeric.isna().any():
                raise ValueError("cannot cast to int while nulls remain")
            self._current_frame()[column] = numeric.astype("int64")
        else:
            self._current_frame()[column] = numeric.astype("float64")

    def _numeric_series(self, column: str) -> pd.Series:
        raw_series = self._current_frame()[column]
        normalized = raw_series.map(self._normalize_numeric_value)
        return pd.to_numeric(normalized, errors="coerce")

    def _normalize_numeric_value(self, value: object) -> object:
        if pd.isna(value):
            return value
        text = str(value).strip()
        text = text.replace(",", "")
        text = re.sub(r"(?i)\b(?:usd|inr|units?)\b", "", text).strip()
        text = text.replace("$", "").replace("₹", "")
        return text

    def _normalize_string_value(self, value: object, column: str | None) -> object:
        if pd.isna(value):
            return value
        text = str(value)
        if "Ã" in text or "Â" in text:
            try:
                text = text.encode("latin1").decode("utf-8")
            except Exception:
                pass
        text = text.strip()
        column_name = (column or "").lower()
        if "date" in column_name:
            parsed_text = self._normalize_date_string(text)
            if parsed_text:
                return parsed_text
        if column_name in {"email", "category", "status"}:
            return text.lower()
        return text

    def _normalize_date_string(self, text: str) -> str | None:
        stripped = text.strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", stripped):
            return stripped
        for fmt in ("%d/%m/%Y", "%m-%d-%Y", "%m/%d/%Y", "%d-%m-%Y"):
            try:
                parsed = datetime.strptime(stripped, fmt)
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                continue
        parsed = pd.to_datetime(stripped, errors="coerce")
        if pd.notna(parsed):
            return parsed.strftime("%Y-%m-%d")
        return None

    def _drop_outliers(self, column: str | None) -> None:
        if column not in self._current_frame().columns:
            raise ValueError("target_column not found")
        selected = self._current_frame()[column]
        if not isinstance(selected, pd.Series):
            raise ValueError("target_column must resolve to a single column")
        numeric = selected.map(self._normalize_numeric_value)
        numeric = pd.to_numeric(numeric, errors="coerce")
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
        if not self._task3_commit_ready():
            return
        orders = self._tables["orders"].copy()
        products = self._tables["products"][["product_id", "unit_price"]].copy()
        if "product_id" not in orders.columns or "product_id" not in products.columns:
            return
        orders["product_id"] = pd.to_numeric(orders["product_id"], errors="coerce")
        products["product_id"] = pd.to_numeric(products["product_id"], errors="coerce")
        merged = orders.merge(products, on="product_id", how="left", suffixes=("", "_product"))
        quantity = pd.to_numeric(merged["quantity"], errors="coerce")
        unit_price = pd.to_numeric(merged["unit_price"], errors="coerce")
        merged["total_price"] = (quantity * unit_price).round(2)
        self._tables["orders"] = merged[orders.columns].copy()

    def _task3_commit_ready(self) -> bool:
        if self._task_id != 3:
            return True

        if self._table_has_structural_issues("customers"):
            return False
        if self._table_has_structural_issues("products"):
            return False
        if self._table_has_structural_issues("orders"):
            return False
        return True

    def _table_has_structural_issues(self, table_name: str) -> bool:
        frame = self._tables[table_name]
        if frame.isnull().sum().sum() > 0:
            return True
        if duplicate_row_count(frame) > 0:
            return True
        if self._schema_report_for_table(table_name):
            return True
        if self._outlier_details_for_frame(frame):
            return True
        if self._format_issue_details_for_frame(frame):
            return True
        return False

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
        duplicate_rate = float(duplicate_row_count(current) / max(len(current), 1))

        schema_report = self._schema_report()
        observation = PipelineDoctorObservation(
            missing_rate=round(missing_rate, 4),
            duplicate_rate=round(duplicate_rate, 4),
            type_violations=len(schema_report),
            outlier_count=self._outlier_count(),
            format_issues=self._format_issue_count(),
            schema_report=schema_report,
            recent_errors=self._recent_errors[:5],
            current_score=self._state.current_score,
            steps_taken=self._state.step_count,
            stage=self._state.active_table,
            available_actions=list(range(16)),
            action_result=action_result,
            table_health=self._table_health(),
            dependency_alerts=self._dependency_alerts(),
            commit_ready=self._task3_commit_ready(),
            scenario_split=self._split,
            reward=reward,
            done=done,
            metadata=metadata or {},
        )
        return observation

    def _schema_report(self) -> dict[str, dict[str, str]]:
        return self._schema_report_for_table(self._state.active_table)

    def _deduplicate_current_table(self) -> pd.DataFrame:
        current = self._current_frame()
        key_column = None
        for candidate in ("transaction_id", "order_id", "customer_id", "product_id"):
            if candidate in current.columns:
                key_column = candidate
                break
        if key_column:
            return current.drop_duplicates(subset=[key_column], keep="first").reset_index(drop=True)
        return current.drop_duplicates().reset_index(drop=True)

    def _schema_report_for_table(self, table_name: str) -> dict[str, dict[str, str]]:
        current = self._tables[table_name]
        expected = self._expected_types[table_name]
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
        return self._outlier_details_for_frame(self._current_frame())

    def _outlier_details_for_frame(self, current: pd.DataFrame) -> dict[str, int]:
        details: dict[str, int] = {}
        for column in current.columns:
            selected = current[column]
            if not isinstance(selected, pd.Series):
                continue
            numeric = pd.to_numeric(selected, errors="coerce")
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

    def _format_issue_details(self) -> dict[str, int]:
        return self._format_issue_details_for_frame(self._current_frame())

    def _format_issue_details_for_frame(self, current: pd.DataFrame) -> dict[str, int]:
        details: dict[str, int] = {}
        for column in current.columns:
            series = current[column]
            if not isinstance(series, pd.Series):
                continue
            issue_count = 0
            column_name = column.lower()
            for value in series.dropna():
                text = str(value)
                normalized = self._normalize_string_value(value, column)
                if "date" in column_name:
                    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(text).strip()):
                        issue_count += 1
                        continue
                elif column_name in {"email", "category", "status"} and text != str(normalized):
                    issue_count += 1
            if issue_count > 0:
                details[column] = issue_count
        return details

    def _format_issue_count(self) -> int:
        return sum(self._format_issue_details().values())

    def _dependency_alerts(self) -> list[str]:
        if self._task_id != 3:
            return []
        alerts: list[str] = []
        mismatch_count = calculation_mismatch_count(
            self._tables["orders"], self._tables["products"]
        )
        if mismatch_count > 0:
            alerts.append(
                "orders.total_price depends on products.unit_price and is still inconsistent"
            )
        if self._table_has_structural_issues("products"):
            alerts.append("products still blocks a safe task 3 commit")
        if self._table_has_structural_issues("orders"):
            alerts.append("orders still contains structural issues")
        return alerts[:3]

    def _table_health(self) -> dict[str, float]:
        if self._task_id != 3:
            return {"single": round(self._state.current_score, 4)}
        return {
            table_name: round(
                score_single_table(
                    self._tables[table_name],
                    self._ground_truth[table_name],
                    self._expected_types[table_name],
                )[0],
                4,
            )
            for table_name in ("orders", "customers", "products")
        }

    def _refresh_errors(self) -> None:
        current = self._current_frame()
        errors: list[str] = []
        null_counts = current.isnull().sum()
        for column, count in null_counts.items():
            if int(count) > 0:
                errors.append(f"{column}: {int(count)} null values")
        duplicate_count = duplicate_row_count(current)
        if duplicate_count > 0:
            errors.append(f"{duplicate_count} duplicate rows detected")
        for column, count in self._outlier_details().items():
            errors.append(f"{column}: {count} outlier values")
        for column, count in self._format_issue_details().items():
            errors.append(f"{column}: {count} format mismatch values")
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
        errors.extend(self._dependency_alerts())
        self._recent_errors = errors[:6]

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
