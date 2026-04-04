from __future__ import annotations

try:
    from ..grading import duplicate_row_count
except ImportError:
    from benchmark.grading import duplicate_row_count


def table_has_structural_issues(env, table_name: str) -> bool:
    frame = env._tables[table_name]
    if frame.isnull().sum().sum() > 0:
        return True
    if duplicate_row_count(frame) > 0:
        return True
    if env._schema_report_for_table(table_name):
        return True
    if env._outlier_details_for_frame(frame):
        return True
    if env._format_issue_details_for_frame(frame):
        return True
    return False
