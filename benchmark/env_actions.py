"""Action execution helpers for the Mario environment."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
import re

import pandas as pd

try:
    from .action_metadata import ACTION_NAMES, PARAMETER_ACTIONS
    from .grading import duplicate_row_count
except ImportError:
    from benchmark.action_metadata import ACTION_NAMES, PARAMETER_ACTIONS
    from benchmark.grading import duplicate_row_count


def apply_action(env, action) -> str:
    if action.action_id not in ACTION_NAMES:
        raise ValueError("unknown action_id")

    if action.action_id == 0:
        return handle_inspect_schema(env, action)
    if action.action_id == 1:
        return "\n".join(env._recent_errors) if env._recent_errors else "No errors detected."
    if action.action_id == 2:
        rows = env._current_table().head(5).to_dict(orient="records")
        return json.dumps(rows, default=str)
    if action.action_id in PARAMETER_ACTIONS and not action.target_column:
        raise ValueError("missing required parameter target_column")
    if action.action_id == 12 and not action.new_name:
        raise ValueError("missing required parameter new_name")
    if action.action_id == 13 and not action.column_order:
        raise ValueError("missing required parameter column_order")

    if action.action_id == 3:
        fill_with_statistic(env, action.target_column, "mean")
    elif action.action_id == 4:
        fill_with_statistic(env, action.target_column, "median")
    elif action.action_id == 5:
        env._current_frame()[action.target_column] = (
            env._current_frame()[action.target_column].ffill().bfill()
        )
    elif action.action_id == 6:
        current = env._current_frame()
        env._tables[env._state.active_table] = current[current[action.target_column].notna()].reset_index(
            drop=True
        )
    elif action.action_id == 7:
        cast_column(env, action.target_column, "int64")
    elif action.action_id == 8:
        cast_column(env, action.target_column, "float64")
    elif action.action_id == 9:
        env._current_frame()[action.target_column] = env._current_frame()[action.target_column].map(
            lambda value: normalize_string_value(env, value, action.target_column)
        )
    elif action.action_id == 10:
        env._tables[env._state.active_table] = deduplicate_current_table(env)
    elif action.action_id == 11:
        drop_outliers(env, action.target_column)
    elif action.action_id == 12:
        if action.target_column not in env._current_frame().columns:
            raise ValueError("target_column not found")
        if action.new_name != action.target_column and action.new_name in env._current_frame().columns:
            raise ValueError("new_name must not duplicate an existing column")
        env._tables[env._state.active_table] = env._current_frame().rename(
            columns={action.target_column: action.new_name}
        )
    elif action.action_id == 13:
        current_columns = list(env._current_frame().columns)
        if Counter(action.column_order) != Counter(current_columns):
            raise ValueError("column_order must match current table columns exactly")
        env._tables[env._state.active_table] = env._current_frame()[action.column_order].copy()
    elif action.action_id == 14:
        return "\n".join(env._recent_errors) if env._recent_errors else "Schema validation passed."
    elif action.action_id == 15:
        return "Changes committed."
    elif action.action_id == 16:
        return scale_resources(env, up=True)
    elif action.action_id == 17:
        return scale_resources(env, up=False)
    elif action.action_id == 18:
        return prioritize_incremental_batch(env)
    elif action.action_id == 19:
        return refresh_downstream_summary(env)

    return ""


def handle_inspect_schema(env, action) -> str:
    if action.target_column and action.target_column in env._tables:
        env._state.active_table = action.target_column
    current = env._current_frame()
    expected = env._expected_types[env._state.active_table]
    lines = [f"table={env._state.active_table}"]
    for column in current.columns:
        lines.append(f"{column}: actual={current[column].dtype}, expected={expected.get(column, 'unknown')}")
    return "\n".join(lines)


def fill_with_statistic(env, column: str | None, statistic: str) -> None:
    if column not in env._current_frame().columns:
        raise ValueError("target_column not found")
    numeric = numeric_series(env, column)
    if numeric.dropna().empty:
        raise ValueError("target_column has no numeric values for statistic fill")
    value = numeric.mean() if statistic == "mean" else numeric.median()
    env._current_frame()[column] = numeric.fillna(value)


def cast_column(env, column: str | None, dtype: str) -> None:
    if column not in env._current_frame().columns:
        raise ValueError("target_column not found")
    numeric = numeric_series(env, column)
    if dtype == "int64":
        if numeric.isna().any():
            raise ValueError("cannot cast to int while nulls remain")
        env._current_frame()[column] = numeric.astype("int64")
    else:
        env._current_frame()[column] = numeric.astype("float64")


def numeric_series(env, column: str) -> pd.Series:
    raw_series = env._current_frame()[column]
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
    if column not in env._current_frame().columns:
        raise ValueError("target_column not found")
    selected = env._current_frame()[column]
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
    env._tables[env._state.active_table] = env._current_frame()[mask].reset_index(drop=True)


def commit_changes(env) -> None:
    if env._task_id == 4 or env._task_id == 5 or env._task_id != 3:
        return
    if not task3_commit_ready(env):
        return
    orders = env._tables["orders"].copy()
    products = env._tables["products"][["product_id", "unit_price"]].copy()
    if "product_id" not in orders.columns or "product_id" not in products.columns:
        return
    orders["product_id"] = pd.to_numeric(orders["product_id"], errors="coerce")
    products["product_id"] = pd.to_numeric(products["product_id"], errors="coerce")
    merged = orders.merge(products, on="product_id", how="left", suffixes=("", "_product"))
    quantity = pd.to_numeric(merged["quantity"], errors="coerce")
    unit_price = pd.to_numeric(merged["unit_price"], errors="coerce")
    merged["total_price"] = (quantity * unit_price).round(2)
    env._tables["orders"] = merged[orders.columns].copy()


def task3_commit_ready(env) -> bool:
    if env._task_id != 3:
        return True
    return not any(
        table_has_structural_issues(env, table_name)
        for table_name in ("customers", "products", "orders")
    )


def task4_commit_ready(env) -> bool:
    if env._task_id != 4:
        return True
    for table_name in ("orders", "products", "daily_summary"):
        if table_has_structural_issues(env, table_name):
            return False
    if env._state.backlog_rows > 0 or env._state.pending_batches > 0:
        return False
    if env._state.freshness_lag_minutes > 0:
        return False
    if bool(env._scenario_meta.get("downstream_stale", False)):
        return False
    return True


def task5_commit_ready(env) -> bool:
    if env._task_id != 5:
        return True
    for table_name in ("source_orders", "catalog", "hourly_rollup"):
        if table_has_structural_issues(env, table_name):
            return False
    if env._state.backlog_rows > 0 or env._state.pending_batches > 0:
        return False
    if env._state.resource_level < env._state.required_resource_level:
        return False
    if env._state.freshness_lag_minutes > 30:
        return False
    if bool(env._scenario_meta.get("downstream_stale", False)):
        return False
    return True


def scale_resources(env, *, up: bool) -> str:
    if env._task_id not in {4, 5}:
        raise ValueError("resource scaling is only available in task 4 or task 5")
    current = env._state.resource_level
    env._state.resource_level = min(current + 1, 3) if up else max(current - 1, 1)
    env._scenario_meta["resource_level"] = env._state.resource_level
    pressure = env._workload_pressure()
    direction = "up" if up else "down"
    return f"resources_scaled_{direction}: level={env._state.resource_level}, pressure={pressure:.2f}"


def prioritize_incremental_batch(env) -> str:
    if env._task_id not in {4, 5}:
        raise ValueError("incremental batch prioritization is only available in task 4 or task 5")
    pending_orders = env._scenario_meta.get("pending_orders")
    if not isinstance(pending_orders, pd.DataFrame) or pending_orders.empty:
        return "No pending incremental batch detected."
    if env._state.resource_level < env._state.required_resource_level:
        raise ValueError("resource level is too low for incremental batch recovery")
    target_table = "orders" if env._task_id == 4 else "source_orders"
    env._tables[target_table] = pd.concat(
        [env._tables[target_table], pending_orders.copy(deep=True)],
        ignore_index=True,
    )
    env._scenario_meta["pending_orders"] = pending_orders.iloc[0:0].copy(deep=True)
    env._state.backlog_rows = 0
    env._state.queue_backlog_age_minutes = 0
    env._state.pending_batches = 0
    env._scenario_meta["backlog_rows"] = 0
    env._scenario_meta["queue_backlog_age_minutes"] = 0
    env._scenario_meta["pending_batches"] = 0
    lag_reduction = 90 if env._task_id == 4 else 120
    env._state.freshness_lag_minutes = max(0, env._state.freshness_lag_minutes - lag_reduction)
    env._scenario_meta["freshness_lag_minutes"] = env._state.freshness_lag_minutes
    return "Incremental batch prioritized and loaded into the live orders table."


def refresh_downstream_summary(env) -> str:
    if env._task_id not in {4, 5}:
        raise ValueError("downstream refresh is only available in task 4 or task 5")
    if env._task_id == 5:
        return refresh_hourly_rollup(env)
    orders = env._tables["orders"].copy()
    products = env._tables["products"].copy()
    if not {"product_id", "quantity", "event_ts"}.issubset(orders.columns):
        raise ValueError("orders table is missing required columns for downstream refresh")
    if not {"product_id", "unit_price"}.issubset(products.columns):
        raise ValueError("products table is missing required columns for downstream refresh")
    orders["product_id"] = pd.to_numeric(orders["product_id"], errors="coerce")
    products["product_id"] = pd.to_numeric(products["product_id"], errors="coerce")
    orders["quantity"] = pd.to_numeric(orders["quantity"], errors="coerce")
    products["unit_price"] = pd.to_numeric(
        products["unit_price"].map(normalize_numeric_value),
        errors="coerce",
    )
    orders["event_date"] = orders["event_ts"].map(
        lambda value: (normalize_date_string(str(value), preserve_time=True) or "")[:10]
    )
    merged = orders.merge(products[["product_id", "unit_price"]], on="product_id", how="left")
    merged["total_revenue"] = merged["quantity"] * merged["unit_price"]
    summary = (
        merged.groupby("event_date", as_index=False)
        .agg(order_count=("order_id", "count"), total_revenue=("total_revenue", "sum"))
    )
    env._tables["daily_summary"] = summary
    env._state.freshness_lag_minutes = 0 if env._state.backlog_rows == 0 else max(
        15, env._state.freshness_lag_minutes
    )
    env._scenario_meta["freshness_lag_minutes"] = env._state.freshness_lag_minutes
    env._scenario_meta["downstream_stale"] = env._state.backlog_rows > 0
    recent_failures = dict(env._scenario_meta.get("recent_failure_counters", {}))
    if "summary_refresh_failures" in recent_failures:
        recent_failures["summary_refresh_failures"] = 0
        env._scenario_meta["recent_failure_counters"] = recent_failures
    return "Downstream daily summary refreshed from the current upstream tables."


def refresh_hourly_rollup(env) -> str:
    source = env._tables["source_orders"].copy()
    catalog = env._tables["catalog"].copy()
    time_column = "event_ts" if "event_ts" in source.columns else "observed_at"
    if not {"product_id", "quantity", time_column}.issubset(source.columns):
        raise ValueError("source_orders is missing required columns for hourly rollup refresh")
    if not {"product_id", "unit_price"}.issubset(catalog.columns):
        raise ValueError("catalog is missing required columns for hourly rollup refresh")
    source["product_id"] = pd.to_numeric(source["product_id"], errors="coerce")
    source["quantity"] = pd.to_numeric(source["quantity"].map(normalize_numeric_value), errors="coerce")
    source["event_ts"] = source[time_column].map(
        lambda value: normalize_date_string(str(value), preserve_time=True)
    )
    catalog["product_id"] = pd.to_numeric(catalog["product_id"], errors="coerce")
    catalog["unit_price"] = pd.to_numeric(
        catalog["unit_price"].map(normalize_numeric_value),
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
    env._tables["hourly_rollup"] = rollup
    env._state.freshness_lag_minutes = 0 if env._state.backlog_rows == 0 else max(
        30, env._state.freshness_lag_minutes
    )
    env._scenario_meta["freshness_lag_minutes"] = env._state.freshness_lag_minutes
    env._scenario_meta["downstream_stale"] = env._state.backlog_rows > 0
    recent_failures = dict(env._scenario_meta.get("recent_failure_counters", {}))
    if "rollup_refresh_failures" in recent_failures:
        recent_failures["rollup_refresh_failures"] = 0
        env._scenario_meta["recent_failure_counters"] = recent_failures
    return "Hourly rollup refreshed from source_orders and catalog."


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


def deduplicate_current_table(env) -> pd.DataFrame:
    current = env._current_frame()
    key_column = None
    for candidate in ("transaction_id", "order_id", "customer_id", "product_id"):
        if candidate in current.columns:
            key_column = candidate
            break
    if key_column:
        return current.drop_duplicates(subset=[key_column], keep="first").reset_index(drop=True)
    return current.drop_duplicates().reset_index(drop=True)
