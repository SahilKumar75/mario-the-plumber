from __future__ import annotations

from benchmark.grading import duplicate_row_count
from benchmark.inspection import format_issue_details_for_frame, outlier_details_for_frame, schema_report_for_table


def table_has_structural_issues(env, table_name: str) -> bool:
    frame = env._tables[table_name]
    if frame.isnull().sum().sum() > 0:
        return True
    if duplicate_row_count(frame) > 0:
        return True
    if schema_report_for_table(env, table_name):
        return True
    # Aggregate summary tables can legitimately contain large bucket-to-bucket variance.
    # For those tables we rely on deterministic scorer checks instead of generic z-score
    # outlier detection, which otherwise blocks safe commits after a correct refresh.
    if table_name not in {"daily_summary", "hourly_rollup"} and outlier_details_for_frame(env, frame):
        return True
    if format_issue_details_for_frame(env, frame):
        return True
    return False
