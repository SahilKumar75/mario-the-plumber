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
    from .data_generator import (
        FORMAL_TASK_SPECS,
        MAX_STEPS,
        TASK_OBJECTIVE_WEIGHTS,
        TASK_THRESHOLDS,
        generate_scenario,
    )
    from .grader import (
        calculation_mismatch_count,
        compute_reward,
        compute_reward_breakdown,
        duplicate_row_count,
        score_single_table,
        score_task3,
        score_task4,
        score_task5,
    )
except ImportError:
    from models import PipelineDoctorAction, PipelineDoctorObservation, PipelineDoctorState
    from server.data_generator import (
        FORMAL_TASK_SPECS,
        MAX_STEPS,
        TASK_OBJECTIVE_WEIGHTS,
        TASK_THRESHOLDS,
        generate_scenario,
    )
    from server.grader import (
        calculation_mismatch_count,
        compute_reward,
        compute_reward_breakdown,
        duplicate_row_count,
        score_single_table,
        score_task3,
        score_task4,
        score_task5,
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
    16: "scale_resources_up",
    17: "scale_resources_down",
    18: "prioritize_incremental_batch",
    19: "refresh_downstream_summary",
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
            freshness_lag_minutes=0,
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
            backlog_rows=int(self._scenario_meta.get("backlog_rows", 0)),
            freshness_lag_minutes=int(self._scenario_meta.get("freshness_lag_minutes", 0)),
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
        elif action.action_id == 16:
            return self._scale_resources(up=True)
        elif action.action_id == 17:
            return self._scale_resources(up=False)
        elif action.action_id == 18:
            return self._prioritize_incremental_batch()
        elif action.action_id == 19:
            return self._refresh_downstream_summary()

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
        if numeric.dropna().empty:
            raise ValueError("target_column has no numeric values for statistic fill")
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
        cents_match = re.fullmatch(r"(?i)\$?\s*(\d+(?:\.\d+)?)\s*cents?", text)
        if cents_match:
            return str(float(cents_match.group(1)) / 100.0)
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
        if self._is_datetime_like_column(column_name):
            parsed_text = self._normalize_date_string(
                text,
                preserve_time=column_name.endswith("_ts") or column_name.endswith("_time"),
            )
            if parsed_text:
                return parsed_text
        if column_name in {"email", "category", "status"}:
            return text.lower()
        return text

    def _normalize_date_string(self, text: str, *, preserve_time: bool = False) -> str | None:
        stripped = text.strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}t\d{2}:\d{2}:\d{2}z", stripped.lower()):
            normalized = stripped.replace("t", "T").replace("z", "Z")
            return normalized if preserve_time else normalized[:10]
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", stripped):
            return stripped
        for fmt in ("%d/%m/%Y %H:%M", "%m-%d-%Y %H:%M"):
            try:
                parsed = datetime.strptime(stripped, fmt)
                return (
                    parsed.strftime("%Y-%m-%dT%H:%M:%SZ")
                    if preserve_time
                    else parsed.strftime("%Y-%m-%d")
                )
            except ValueError:
                continue
        for fmt in ("%d/%m/%Y", "%m-%d-%Y", "%m/%d/%Y", "%d-%m-%Y"):
            try:
                parsed = datetime.strptime(stripped, fmt)
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                continue
        parsed = pd.to_datetime(stripped, errors="coerce")
        if pd.notna(parsed):
            if self._looks_timestamp_string(stripped):
                parsed = pd.to_datetime(stripped, errors="coerce", utc=True)
                return (
                    parsed.strftime("%Y-%m-%dT%H:%M:%SZ")
                    if preserve_time
                    else parsed.strftime("%Y-%m-%d")
                )
            return parsed.strftime("%Y-%m-%d")
        return None

    def _is_datetime_like_column(self, column_name: str) -> bool:
        return "date" in column_name or column_name.endswith("_ts") or column_name.endswith("_time")

    def _looks_timestamp_string(self, text: str) -> bool:
        return bool(re.search(r"\d{2}:\d{2}", text))

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
        if self._task_id == 4:
            return
        if self._task_id == 5:
            return
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

    def _task4_commit_ready(self) -> bool:
        if self._task_id != 4:
            return True
        for table_name in ("orders", "products", "daily_summary"):
            if self._table_has_structural_issues(table_name):
                return False
        if self._state.backlog_rows > 0 or self._state.pending_batches > 0:
            return False
        if self._state.freshness_lag_minutes > 0:
            return False
        if bool(self._scenario_meta.get("downstream_stale", False)):
            return False
        return True

    def _task5_commit_ready(self) -> bool:
        if self._task_id != 5:
            return True
        for table_name in ("source_orders", "catalog", "hourly_rollup"):
            if self._table_has_structural_issues(table_name):
                return False
        if self._state.backlog_rows > 0 or self._state.pending_batches > 0:
            return False
        if self._state.resource_level < self._state.required_resource_level:
            return False
        if self._state.freshness_lag_minutes > 30:
            return False
        if bool(self._scenario_meta.get("downstream_stale", False)):
            return False
        return True

    def _scale_resources(self, *, up: bool) -> str:
        if self._task_id not in {4, 5}:
            raise ValueError("resource scaling is only available in task 4 or task 5")
        current = self._state.resource_level
        if up:
            self._state.resource_level = min(current + 1, 3)
        else:
            self._state.resource_level = max(current - 1, 1)
        self._scenario_meta["resource_level"] = self._state.resource_level
        pressure = self._workload_pressure()
        direction = "up" if up else "down"
        return f"resources_scaled_{direction}: level={self._state.resource_level}, pressure={pressure:.2f}"

    def _prioritize_incremental_batch(self) -> str:
        if self._task_id not in {4, 5}:
            raise ValueError("incremental batch prioritization is only available in task 4 or task 5")
        pending_orders = self._scenario_meta.get("pending_orders")
        if not isinstance(pending_orders, pd.DataFrame) or pending_orders.empty:
            return "No pending incremental batch detected."
        if self._state.resource_level < self._state.required_resource_level:
            raise ValueError("resource level is too low for incremental batch recovery")
        target_table = "orders" if self._task_id == 4 else "source_orders"
        self._tables[target_table] = pd.concat(
            [self._tables[target_table], pending_orders.copy(deep=True)],
            ignore_index=True,
        )
        self._scenario_meta["pending_orders"] = pending_orders.iloc[0:0].copy(deep=True)
        self._state.backlog_rows = 0
        self._state.pending_batches = 0
        self._scenario_meta["backlog_rows"] = 0
        self._scenario_meta["pending_batches"] = 0
        lag_reduction = 90 if self._task_id == 4 else 120
        self._state.freshness_lag_minutes = max(0, self._state.freshness_lag_minutes - lag_reduction)
        self._scenario_meta["freshness_lag_minutes"] = self._state.freshness_lag_minutes
        return "Incremental batch prioritized and loaded into the live orders table."

    def _refresh_downstream_summary(self) -> str:
        if self._task_id not in {4, 5}:
            raise ValueError("downstream refresh is only available in task 4 or task 5")
        if self._task_id == 5:
            return self._refresh_hourly_rollup()
        orders = self._tables["orders"].copy()
        products = self._tables["products"].copy()
        if not {"product_id", "quantity", "event_ts"}.issubset(orders.columns):
            raise ValueError("orders table is missing required columns for downstream refresh")
        if not {"product_id", "unit_price"}.issubset(products.columns):
            raise ValueError("products table is missing required columns for downstream refresh")
        orders["product_id"] = pd.to_numeric(orders["product_id"], errors="coerce")
        products["product_id"] = pd.to_numeric(products["product_id"], errors="coerce")
        orders["quantity"] = pd.to_numeric(orders["quantity"], errors="coerce")
        products["unit_price"] = pd.to_numeric(
            products["unit_price"].map(self._normalize_numeric_value),
            errors="coerce",
        )
        orders["event_date"] = orders["event_ts"].map(
            lambda value: (
                self._normalize_date_string(str(value), preserve_time=True) or ""
            )[:10]
        )
        merged = orders.merge(products[["product_id", "unit_price"]], on="product_id", how="left")
        merged["total_revenue"] = merged["quantity"] * merged["unit_price"]
        summary = (
            merged.groupby("event_date", as_index=False)
            .agg(order_count=("order_id", "count"), total_revenue=("total_revenue", "sum"))
        )
        self._tables["daily_summary"] = summary
        self._state.freshness_lag_minutes = 0 if self._state.backlog_rows == 0 else max(15, self._state.freshness_lag_minutes)
        self._scenario_meta["freshness_lag_minutes"] = self._state.freshness_lag_minutes
        self._scenario_meta["downstream_stale"] = self._state.backlog_rows > 0
        return "Downstream daily summary refreshed from the current upstream tables."

    def _refresh_hourly_rollup(self) -> str:
        source = self._tables["source_orders"].copy()
        catalog = self._tables["catalog"].copy()
        time_column = "event_ts" if "event_ts" in source.columns else "observed_at"
        if not {"product_id", "quantity", time_column}.issubset(source.columns):
            raise ValueError("source_orders is missing required columns for hourly rollup refresh")
        if not {"product_id", "unit_price"}.issubset(catalog.columns):
            raise ValueError("catalog is missing required columns for hourly rollup refresh")
        source["product_id"] = pd.to_numeric(source["product_id"], errors="coerce")
        source["quantity"] = pd.to_numeric(source["quantity"].map(self._normalize_numeric_value), errors="coerce")
        source["event_ts"] = source[time_column].map(
            lambda value: self._normalize_date_string(str(value), preserve_time=True)
        )
        catalog["product_id"] = pd.to_numeric(catalog["product_id"], errors="coerce")
        catalog["unit_price"] = pd.to_numeric(
            catalog["unit_price"].map(self._normalize_numeric_value),
            errors="coerce",
        )
        merged = source.merge(catalog[["product_id", "unit_price"]], on="product_id", how="left")
        merged["gross_revenue"] = merged["quantity"] * merged["unit_price"]
        merged["hour_bucket"] = pd.to_datetime(merged["event_ts"], errors="coerce", utc=True).dt.strftime(
            "%Y-%m-%dT%H:00:00Z"
        )
        rollup = (
            merged.groupby("hour_bucket", as_index=False)
            .agg(order_count=("order_id", "count"), gross_revenue=("gross_revenue", "sum"))
        )
        self._tables["hourly_rollup"] = rollup
        self._state.freshness_lag_minutes = 0 if self._state.backlog_rows == 0 else max(30, self._state.freshness_lag_minutes)
        self._scenario_meta["freshness_lag_minutes"] = self._state.freshness_lag_minutes
        self._scenario_meta["downstream_stale"] = self._state.backlog_rows > 0
        return "Hourly rollup refreshed from source_orders and catalog."

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
        alias_hints = self._column_alias_hints()
        subgoal_progress, subgoal_order, active_subgoal, reward_machine_state = self._task_progress_bundle()
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
            available_actions=list(range(20)),
            action_result=action_result,
            table_health=self._table_health(),
            dependency_alerts=self._dependency_alerts(),
            commit_ready=self._commit_ready(),
            scenario_split=self._split,
            schema_drift_count=len(schema_report) + self._format_issue_count(),
            backlog_rows=self._state.backlog_rows,
            freshness_lag_minutes=self._state.freshness_lag_minutes,
            resource_level=self._state.resource_level,
            required_resource_level=self._state.required_resource_level,
            workload_pressure=self._workload_pressure(),
            pending_batches=self._state.pending_batches,
            downstream_stale=bool(self._scenario_meta.get("downstream_stale", False)),
            orchestration_alerts=self._orchestration_alerts(),
            observed_columns=list(current.columns),
            missing_expected_columns=self._missing_expected_columns(self._state.active_table),
            column_alias_hints=alias_hints,
            scenario_profile=str(self._scenario_meta.get("scenario_profile", "baseline")),
            open_world_patterns=list(self._scenario_meta.get("open_world_patterns", [])),
            time_budget_remaining=max(0, self._state.max_steps - self._state.step_count),
            time_budget_ratio=round(
                max(0.0, (self._state.max_steps - self._state.step_count) / max(self._state.max_steps, 1)),
                4,
            ),
            truncated=self._state.truncated,
            done_reason=self._state.done_reason,
            synthetic_data_notes=list(self._scenario_meta.get("synthetic_data_notes", [])),
            reward_breakdown=dict(self._last_reward_breakdown),
            objective_breakdown=self._objective_breakdown(),
            tradeoff_weights=dict(TASK_OBJECTIVE_WEIGHTS.get(self._task_id, {})),
            subgoal_progress=subgoal_progress,
            subgoal_order=subgoal_order,
            active_subgoal=active_subgoal,
            reward_machine_state=reward_machine_state,
            adaptation_target=str(self._scenario_meta.get("adaptation_target", "")),
            heldout_profile_family=bool(self._scenario_meta.get("heldout_profile_family", False)),
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

    def _missing_expected_columns(self, table_name: str) -> list[str]:
        current = self._tables[table_name]
        expected = self._expected_types[table_name]
        return [column for column in expected if column not in current.columns]

    def _column_alias_hints(self) -> dict[str, str]:
        current_columns = set(self._current_frame().columns)
        aliases = {
            "product_category": "category",
            "product_segment": "category",
            "event_time": "event_ts",
            "business_date": "event_date",
            "observed_at": "event_ts",
            "window_start": "hour_bucket",
        }
        hints: dict[str, str] = {}
        for drifted_name, expected_name in aliases.items():
            if drifted_name in current_columns and expected_name not in current_columns:
                hints[drifted_name] = expected_name
        return hints

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
                if self._is_datetime_like_column(column_name):
                    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(text).strip()):
                        if column_name.endswith("_ts") or column_name.endswith("_time"):
                            if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", str(text).strip()):
                                continue
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
        if self._task_id == 5:
            alerts: list[str] = []
            if self._state.backlog_rows > 0:
                alerts.append("late source batches still need replay into source_orders")
            if bool(self._scenario_meta.get("downstream_stale", False)):
                alerts.append("hourly_rollup is stale relative to source_orders and catalog")
            if self._state.freshness_lag_minutes > 30:
                alerts.append("freshness SLA is still violated for the temporal pipeline")
            if self._state.resource_level < self._state.required_resource_level and self._state.backlog_rows > 0:
                alerts.append("resource level is too low for late-batch replay")
            return alerts[:4]
        if self._task_id == 4:
            alerts: list[str] = []
            if self._state.backlog_rows > 0:
                alerts.append("latest incremental batch is still pending in the orders stream")
            if bool(self._scenario_meta.get("downstream_stale", False)):
                alerts.append("daily_summary is stale relative to upstream orders/products")
            if self._state.resource_level < self._state.required_resource_level and self._state.backlog_rows > 0:
                alerts.append("resource level is too low for backlog recovery")
            return alerts[:3]
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
        if self._task_id == 5:
            return {
                table_name: round(
                    score_single_table(
                        self._tables[table_name],
                        self._ground_truth[table_name],
                        self._expected_types[table_name],
                    )[0],
                    4,
                )
                for table_name in ("source_orders", "catalog", "hourly_rollup")
            }
        if self._task_id == 4:
            return {
                table_name: round(
                    score_single_table(
                        self._tables[table_name],
                        self._ground_truth[table_name],
                        self._expected_types[table_name],
                    )[0],
                    4,
                )
                for table_name in ("orders", "products", "daily_summary")
            }
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
        if self._task_id == 4:
            if self._state.backlog_rows > 0:
                errors.append(f"backlog: {self._state.backlog_rows} pending rows still need ingestion")
            if self._state.pending_batches > 0:
                errors.append(f"pending_batches: {self._state.pending_batches} incremental batch remains unprocessed")
            if bool(self._scenario_meta.get("downstream_stale", False)):
                errors.append("daily_summary: downstream aggregate is stale")
            if self._state.resource_level < self._state.required_resource_level and self._state.backlog_rows > 0:
                errors.append(
                    f"resources: level {self._state.resource_level} below required {self._state.required_resource_level}"
                )
        if self._task_id == 5:
            if self._state.backlog_rows > 0:
                errors.append(f"late_batches: {self._state.backlog_rows} rows still await replay")
            if self._state.pending_batches > 0:
                errors.append(f"pending_batches: {self._state.pending_batches} temporal batches remain unreplayed")
            if bool(self._scenario_meta.get("downstream_stale", False)):
                errors.append("hourly_rollup: downstream aggregate is stale")
            if self._state.freshness_lag_minutes > 30:
                errors.append(
                    f"freshness_sla: lag is {self._state.freshness_lag_minutes} minutes"
                )
            if self._state.resource_level < self._state.required_resource_level and self._state.backlog_rows > 0:
                errors.append(
                    f"resources: level {self._state.resource_level} below temporal requirement {self._state.required_resource_level}"
                )
        errors.extend(self._dependency_alerts())
        errors.extend(self._orchestration_alerts())
        self._recent_errors = errors[:6]

    def _score(self) -> float:
        if self._task_id == 3:
            score, _ = score_task3(self._tables, self._ground_truth, self._expected_types)
            return score
        if self._task_id == 4:
            score, _ = score_task4(
                self._tables,
                self._ground_truth,
                self._expected_types,
                backlog_rows=self._state.backlog_rows,
                freshness_lag_minutes=self._state.freshness_lag_minutes,
                resource_level=self._state.resource_level,
                required_resource_level=self._state.required_resource_level,
                downstream_stale=bool(self._scenario_meta.get("downstream_stale", False)),
            )
            return score
        if self._task_id == 5:
            score, _ = score_task5(
                self._tables,
                self._ground_truth,
                self._expected_types,
                backlog_rows=self._state.backlog_rows,
                freshness_lag_minutes=self._state.freshness_lag_minutes,
                resource_level=self._state.resource_level,
                required_resource_level=self._state.required_resource_level,
                downstream_stale=bool(self._scenario_meta.get("downstream_stale", False)),
            )
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
            "truncated": self._state.truncated,
            "done_reason": self._state.done_reason,
            "scenario_profile": self._state.scenario_profile,
        }
        EPISODE_SUMMARIES[self._state.episode_id or str(uuid4())] = summary

    def _breakdown_payload(self) -> dict[str, object]:
        if self._task_id == 3:
            _, breakdown = score_task3(self._tables, self._ground_truth, self._expected_types)
            return breakdown
        if self._task_id == 4:
            _, breakdown = score_task4(
                self._tables,
                self._ground_truth,
                self._expected_types,
                backlog_rows=self._state.backlog_rows,
                freshness_lag_minutes=self._state.freshness_lag_minutes,
                resource_level=self._state.resource_level,
                required_resource_level=self._state.required_resource_level,
                downstream_stale=bool(self._scenario_meta.get("downstream_stale", False)),
            )
            return breakdown
        if self._task_id == 5:
            _, breakdown = score_task5(
                self._tables,
                self._ground_truth,
                self._expected_types,
                backlog_rows=self._state.backlog_rows,
                freshness_lag_minutes=self._state.freshness_lag_minutes,
                resource_level=self._state.resource_level,
                required_resource_level=self._state.required_resource_level,
                downstream_stale=bool(self._scenario_meta.get("downstream_stale", False)),
            )
            return breakdown
        _, breakdown = score_single_table(
            self._tables["single"],
            self._ground_truth["single"],
            self._expected_types["single"],
        )
        return breakdown

    def _workload_pressure(self) -> float:
        base_pressure = float(self._scenario_meta.get("workload_pressure", 0.0))
        if self._task_id not in {4, 5}:
            return round(base_pressure, 4)
        backlog_bonus = min(self._state.backlog_rows / 10.0, 0.4 if self._task_id == 4 else 0.5)
        resource_relief = 0.15 * max(self._state.resource_level - 1, 0)
        pressure = max(0.0, min(1.0, base_pressure + backlog_bonus - resource_relief))
        return round(pressure, 4)

    def _orchestration_alerts(self) -> list[str]:
        if self._task_id == 5:
            alerts: list[str] = []
            if self._state.backlog_rows > 0 and self._state.resource_level < self._state.required_resource_level:
                alerts.append("scale resources before replaying the held-out temporal batches")
            if self._state.backlog_rows == 0 and bool(self._scenario_meta.get("downstream_stale", False)):
                alerts.append("refresh hourly_rollup after replay to close the temporal task")
            if self._state.freshness_lag_minutes > 30:
                alerts.append("bring freshness lag below the 30-minute SLA before committing")
            return alerts[:3]
        if self._task_id != 4:
            return []
        alerts: list[str] = []
        if self._state.backlog_rows > 0 and self._state.resource_level < self._state.required_resource_level:
            alerts.append("scale resources up before prioritizing the delayed batch")
        if self._state.backlog_rows == 0 and bool(self._scenario_meta.get("downstream_stale", False)):
            alerts.append("refresh daily_summary before committing the recovery")
        if self._state.freshness_lag_minutes > 0:
            alerts.append(
                f"freshness lag is still {self._state.freshness_lag_minutes} minutes"
            )
        return alerts[:3]

    def _commit_ready(self) -> bool:
        if self._task_id == 3:
            return self._task3_commit_ready()
        if self._task_id == 4:
            return self._task4_commit_ready()
        if self._task_id == 5:
            return self._task5_commit_ready()
        return True

    def _objective_breakdown(self) -> dict[str, float]:
        breakdown = self._breakdown_payload()
        pipeline = breakdown.get("pipeline", {}) if isinstance(breakdown, dict) else {}
        return {
            key: round(float(value), 4)
            for key, value in pipeline.items()
            if isinstance(value, (int, float))
        }

    def _task_progress_bundle(self) -> tuple[dict[str, bool], list[str], str, str]:
        subgoal_progress = self._subgoal_progress()
        order = list(FORMAL_TASK_SPECS.get(self._task_id, {}).get("reward_machine_order", []))
        active_subgoal = next((name for name in order if not subgoal_progress.get(name, False)), "")
        completed = sum(1 for name in order if subgoal_progress.get(name, False))
        reward_machine_state = (
            f"s{completed}/{len(order)}:{active_subgoal or 'terminal'}" if order else ""
        )
        return subgoal_progress, order, active_subgoal, reward_machine_state

    def _update_task_progress_state(self) -> None:
        _, _, active_subgoal, reward_machine_state = self._task_progress_bundle()
        self._state.active_subgoal = active_subgoal
        self._state.reward_machine_state = reward_machine_state

    def _subgoal_progress(self) -> dict[str, bool]:
        if self._task_id == 3:
            progress = {
                "repair_customers": not self._table_has_structural_issues("customers"),
                "repair_products": not self._table_has_structural_issues("products"),
                "repair_orders": not self._table_has_structural_issues("orders"),
                "restore_dependency_consistency": calculation_mismatch_count(
                    self._tables["orders"],
                    self._tables["products"],
                ) == 0,
                "commit_pipeline": bool(self._state.done and self._state.success),
            }
            return progress
        if self._task_id == 4:
            return {
                "normalize_orders_stream": not self._table_has_structural_issues("orders"),
                "scale_resources_if_needed": (
                    self._state.resource_level >= self._state.required_resource_level
                    if self._state.backlog_rows > 0
                    else True
                ),
                "load_incremental_backlog": self._state.backlog_rows == 0 and self._state.pending_batches == 0,
                "refresh_daily_summary": (
                    not bool(self._scenario_meta.get("downstream_stale", False))
                    and self._state.freshness_lag_minutes == 0
                ),
                "commit_recovery": bool(self._state.done and self._state.success),
            }
        if self._task_id == 5:
            schema_ok = not self._table_has_structural_issues("source_orders") and not self._table_has_structural_issues("catalog")
            no_aliases = not self._missing_expected_columns("source_orders") and not self._missing_expected_columns("catalog")
            return {
                "reconcile_schema_aliases": no_aliases,
                "repair_catalog_and_source_quality": schema_ok,
                "replay_late_batches": self._state.backlog_rows == 0 and self._state.pending_batches == 0,
                "refresh_temporal_rollup": (
                    not self._table_has_structural_issues("hourly_rollup")
                    and not bool(self._scenario_meta.get("downstream_stale", False))
                ),
                "meet_freshness_sla": self._state.freshness_lag_minutes <= 30,
                "commit_temporal_pipeline": bool(self._state.done and self._state.success),
            }
        return {}
