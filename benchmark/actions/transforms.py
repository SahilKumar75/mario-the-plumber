from __future__ import annotations

from collections import Counter
from datetime import datetime
import re

import pandas as pd

from benchmark.runtime_state import current_frame


def handle_inspect_schema(env, action) -> str:
    if action.target_column and action.target_column in env._tables:
        env._state.active_table = action.target_column
    current = current_frame(env)
    expected = env._expected_types[env._state.active_table]
    lines = [f"table={env._state.active_table}"]
    for column in current.columns:
        lines.append(f"{column}: actual={current[column].dtype}, expected={expected.get(column, 'unknown')}")
    return "\n".join(lines)


def fill_with_statistic(env, column: str | None, statistic: str) -> None:
    if column not in current_frame(env).columns:
        raise ValueError("target_column not found")
    numeric = numeric_series(env, column)
    if numeric.dropna().empty:
        raise ValueError("target_column has no numeric values for statistic fill")
    value = numeric.mean() if statistic == "mean" else numeric.median()
    current_frame(env)[column] = numeric.fillna(value)


def cast_column(env, column: str | None, dtype: str) -> None:
    if column not in current_frame(env).columns:
        raise ValueError("target_column not found")
    numeric = numeric_series(env, column)
    if dtype == "int64":
        if numeric.isna().any():
            raise ValueError("cannot cast to int while nulls remain")
        current_frame(env)[column] = numeric.astype("int64")
    else:
        current_frame(env)[column] = numeric.astype("float64")


def numeric_series(env, column: str) -> pd.Series:
    raw_series = current_frame(env)[column]
    normalized = raw_series.map(normalize_numeric_value)
    return pd.to_numeric(normalized, errors="coerce")


def normalize_numeric_value(value: object) -> object:
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


def normalize_string_value(env, value: object, column: str | None) -> object:
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
    if is_datetime_like_column(column_name):
        parsed_text = normalize_date_string(
            text,
            preserve_time=column_name.endswith("_ts") or column_name.endswith("_time"),
        )
        if parsed_text:
            return parsed_text
    if column_name in {"email", "category", "status"}:
        return text.lower()
    return text


def normalize_date_string(text: str, *, preserve_time: bool = False) -> str | None:
    stripped = text.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}t\d{2}:\d{2}:\d{2}z", stripped.lower()):
        normalized = stripped.replace("t", "T").replace("z", "Z")
        return normalized if preserve_time else normalized[:10]
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", stripped):
        return stripped
    for fmt in ("%d/%m/%Y %H:%M", "%m-%d-%Y %H:%M"):
        try:
            parsed = datetime.strptime(stripped, fmt)
            return parsed.strftime("%Y-%m-%dT%H:%M:%SZ") if preserve_time else parsed.strftime("%Y-%m-%d")
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
        if looks_timestamp_string(stripped):
            parsed = pd.to_datetime(stripped, errors="coerce", utc=True)
            return parsed.strftime("%Y-%m-%dT%H:%M:%SZ") if preserve_time else parsed.strftime("%Y-%m-%d")
        return parsed.strftime("%Y-%m-%d")
    return None


def is_datetime_like_column(column_name: str) -> bool:
    return "date" in column_name or column_name.endswith("_ts") or column_name.endswith("_time")


def looks_timestamp_string(text: str) -> bool:
    return bool(re.search(r"\d{2}:\d{2}", text))


def drop_outliers(env, column: str | None) -> None:
    if column not in current_frame(env).columns:
        raise ValueError("target_column not found")
    selected = current_frame(env)[column]
    if not isinstance(selected, pd.Series):
        raise ValueError("target_column must resolve to a single column")
    numeric = selected.map(normalize_numeric_value)
    numeric = pd.to_numeric(numeric, errors="coerce")
    std = float(numeric.std(skipna=True))
    if pd.isna(std) or std == 0:
        return
    mean = float(numeric.mean(skipna=True))
    mask = (numeric - mean).abs() <= 3 * std
    mask = mask.fillna(False)
    env._tables[env._state.active_table] = current_frame(env)[mask].reset_index(drop=True)


def validate_parameter_action(action, current_columns: list[str]) -> None:
    if action.action_id == 12 and not action.new_name:
        raise ValueError("missing required parameter new_name")
    if action.action_id == 13 and not action.column_order:
        raise ValueError("missing required parameter column_order")
    if action.action_id == 12:
        if action.target_column not in current_columns:
            raise ValueError("target_column not found")
        if action.new_name != action.target_column and action.new_name in current_columns:
            raise ValueError("new_name must not duplicate an existing column")
    if action.action_id == 13 and Counter(action.column_order) != Counter(current_columns):
        raise ValueError("column_order must match current table columns exactly")


def deduplicate_current_table(env) -> pd.DataFrame:
    current = current_frame(env)
    key_column = None
    for candidate in ("transaction_id", "order_id", "customer_id", "product_id"):
        if candidate in current.columns:
            key_column = candidate
            break
    if key_column:
        return current.drop_duplicates(subset=[key_column], keep="first").reset_index(drop=True)
    return current.drop_duplicates().reset_index(drop=True)
