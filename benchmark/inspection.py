from __future__ import annotations

try:
    from .grading import calculation_mismatch_count, duplicate_row_count
    from .runtime_state import current_frame
except ImportError:
    from benchmark.grading import calculation_mismatch_count, duplicate_row_count
    from benchmark.runtime_state import current_frame


def schema_report(env) -> dict[str, dict[str, str]]:
    return schema_report_for_table(env, env._state.active_table)


def schema_report_for_table(env, table_name: str) -> dict[str, dict[str, str]]:
    current = env._tables[table_name]
    expected = env._expected_types[table_name]
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


def outlier_details_for_frame(env, current) -> dict[str, int]:
    details: dict[str, int] = {}
    for column in current.columns:
        selected = current[column]
        numeric = current[column] if hasattr(selected, "dtype") else None
        if numeric is None:
            continue
        numeric = env_numeric_series(env, column, current=current)
        if numeric.isna().all():
            continue
        std = float(numeric.std(skipna=True))
        if std == 0 or std != std:
            continue
        mean = float(numeric.mean(skipna=True))
        count = int(((numeric - mean).abs() > 3 * std).fillna(False).sum())
        if count > 0:
            details[column] = count
    return details


def outlier_details(env) -> dict[str, int]:
    return outlier_details_for_frame(env, current_frame(env))


def outlier_count(env) -> int:
    return sum(outlier_details(env).values())


def format_issue_details_for_frame(env, current) -> dict[str, int]:
    details: dict[str, int] = {}
    for column in current.columns:
        series = current[column]
        if not hasattr(series, "dropna"):
            continue
        issue_count = 0
        column_name = column.lower()
        for value in series.dropna():
            text = str(value)
            normalized = env_normalize_string_value(env, value, column)
            if env_is_datetime_like_column(column_name):
                if env_datetime_matches_canonical(text, column_name):
                    continue
                issue_count += 1
                continue
            if column_name in {"email", "category", "status"} and text != str(normalized):
                issue_count += 1
        if issue_count > 0:
            details[column] = issue_count
    return details


def format_issue_details(env) -> dict[str, int]:
    return format_issue_details_for_frame(env, current_frame(env))


def format_issue_count(env) -> int:
    return sum(format_issue_details(env).values())


def env_numeric_series(env, column: str, *, current=None):
    from .actions.transforms import numeric_series

    if current is None or current is current_frame(env):
        return numeric_series(env, column)
    normalized = current[column].map(env_normalize_numeric_value)
    return __import__("pandas").to_numeric(normalized, errors="coerce")


def env_normalize_numeric_value(value: object) -> object:
    from .actions.transforms import normalize_numeric_value

    return normalize_numeric_value(value)


def env_normalize_string_value(env, value: object, column: str | None) -> object:
    from .actions.transforms import normalize_string_value

    return normalize_string_value(env, value, column)


def env_is_datetime_like_column(column_name: str) -> bool:
    from .actions.transforms import is_datetime_like_column

    return is_datetime_like_column(column_name)


def env_datetime_matches_canonical(text: str, column_name: str) -> bool:
    import re

    stripped = str(text).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", stripped):
        return True
    if column_name.endswith("_ts") or column_name.endswith("_time"):
        return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", stripped))
    return False


def structural_mismatch_errors(env) -> list[str]:
    current = current_frame(env)
    errors: list[str] = []
    null_counts = current.isnull().sum()
    for column, count in null_counts.items():
        if int(count) > 0:
            errors.append(f"{column}: {int(count)} null values")
    duplicate_count = duplicate_row_count(current)
    if duplicate_count > 0:
        errors.append(f"{duplicate_count} duplicate rows detected")
    for column, count in outlier_details(env).items():
        errors.append(f"{column}: {count} outlier values")
    for column, count in format_issue_details(env).items():
        errors.append(f"{column}: {count} format mismatch values")
    for column, info in schema_report(env).items():
        errors.append(f"{column}: expected {info['expected']}, found {info['actual']}")
    if env._task_id == 3 and env._state.active_table == "orders":
        mismatch_count = calculation_mismatch_count(env._tables["orders"], env._tables["products"])
        if mismatch_count > 0:
            errors.append(f"total_price: {mismatch_count} rows have calculation mismatch")
    return errors
