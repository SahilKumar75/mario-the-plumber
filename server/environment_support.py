from __future__ import annotations

import pandas as pd

try:
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
    from ..models import PipelineDoctorAction, PipelineDoctorObservation
except ImportError:
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
    from models import PipelineDoctorAction, PipelineDoctorObservation


class EnvironmentSupportMixin:
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
        self.EPISODE_SUMMARIES = self._episode_summaries
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
