from __future__ import annotations

import pandas as pd

try:
    from .transforms import normalize_date_string, normalize_numeric_value
    from .validation import table_has_structural_issues
    from ..grading import task3_dependency_score, task5_rollup_consistency_score, task5_temporal_closure_score
    from ..observation_support import workload_pressure
except ImportError:
    from benchmark.actions.transforms import normalize_date_string, normalize_numeric_value
    from benchmark.actions.validation import table_has_structural_issues
    from benchmark.grading import task3_dependency_score, task5_rollup_consistency_score, task5_temporal_closure_score
    from benchmark.observation_support import workload_pressure


def commit_changes(env) -> None:
    if env._task_id != 3:
        return
    if not task3_commit_ready(env):
        return


def task3_commit_ready(env) -> bool:
    if env._task_id != 3:
        return True
    if any(
        table_has_structural_issues(env, table_name)
        for table_name in ("customers", "products", "orders")
    ):
        return False
    return task3_dependency_score(env._tables["orders"], env._tables["products"]) >= 0.9999


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
    return (
        task5_rollup_consistency_score(
            env._tables["source_orders"],
            env._tables["catalog"],
            env._tables["hourly_rollup"],
        ) >= 0.9999
        and task5_temporal_closure_score(
            env._tables["source_orders"],
            env._tables["hourly_rollup"],
            env._scenario_meta.get("incident_manifest"),
        ) >= 0.9999
    )


def scale_resources(env, *, up: bool) -> str:
    if env._task_id not in {4, 5}:
        raise ValueError("resource scaling is only available in task 4 or task 5")
    current = env._state.resource_level
    env._state.resource_level = min(current + 1, 3) if up else max(current - 1, 1)
    env._scenario_meta["resource_level"] = env._state.resource_level
    pressure = workload_pressure(env)
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
    if "batch_id" not in pending_orders.columns:
        raise ValueError("pending incremental batch metadata is missing batch identifiers")

    batch_ids = [str(value) for value in pending_orders["batch_id"].dropna().astype(str).tolist()]
    if not batch_ids:
        return "No pending incremental batch detected."
    prioritized_batch = sorted(dict.fromkeys(batch_ids))[0]
    replay_batch = pending_orders[pending_orders["batch_id"].astype(str) == prioritized_batch].copy(deep=True)
    remaining_orders = pending_orders[pending_orders["batch_id"].astype(str) != prioritized_batch].copy(deep=True)
    target_table = "orders" if env._task_id == 4 else "source_orders"
    env._tables[target_table] = pd.concat(
        [env._tables[target_table], replay_batch],
        ignore_index=True,
    )
    env._scenario_meta["pending_orders"] = remaining_orders
    env._state.backlog_rows = int(len(remaining_orders))
    env._state.pending_batches = int(remaining_orders["batch_id"].nunique()) if not remaining_orders.empty else 0
    env._scenario_meta["backlog_rows"] = env._state.backlog_rows
    env._scenario_meta["pending_batches"] = env._state.pending_batches
    env._state.queue_backlog_age_minutes = (
        0 if remaining_orders.empty else max(30, env._state.queue_backlog_age_minutes - (90 if env._task_id == 4 else 120))
    )
    env._scenario_meta["queue_backlog_age_minutes"] = env._state.queue_backlog_age_minutes
    lag_reduction = 45 if env._task_id == 4 else 60
    env._state.freshness_lag_minutes = max(0, env._state.freshness_lag_minutes - lag_reduction)
    env._scenario_meta["freshness_lag_minutes"] = env._state.freshness_lag_minutes
    env._scenario_meta["last_replayed_batch_id"] = prioritized_batch
    return (
        f"Incremental batch {prioritized_batch} prioritized and replayed; "
        f"{env._state.backlog_rows} rows remain across {env._state.pending_batches} pending batches."
    )


def refresh_downstream_summary(env) -> str:
    if env._task_id == 3:
        orders = env._tables["orders"].copy()
        products = env._tables["products"].copy()
        if not {"product_id", "quantity", "total_price"}.issubset(orders.columns):
            raise ValueError("orders table is missing required columns for dependency refresh")
        if not {"product_id", "unit_price"}.issubset(products.columns):
            raise ValueError("products table is missing required columns for dependency refresh")
        orders["product_id"] = pd.to_numeric(orders["product_id"], errors="coerce")
        orders["quantity"] = pd.to_numeric(orders["quantity"], errors="coerce")
        products["product_id"] = pd.to_numeric(products["product_id"], errors="coerce")
        products["unit_price"] = pd.to_numeric(
            products["unit_price"].map(normalize_numeric_value),
            errors="coerce",
        )
        merged = orders.merge(products[["product_id", "unit_price"]], on="product_id", how="left")
        merged["total_price"] = (merged["quantity"] * merged["unit_price"]).round(2)
        env._tables["orders"] = merged[orders.columns].copy()
        return "Order totals refreshed from the repaired product catalog."
    if env._task_id not in {4, 5}:
        raise ValueError("downstream refresh is only available in task 3, task 4, or task 5")
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
