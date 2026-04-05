from __future__ import annotations

import pandas as pd

from .shared import IncidentFixture


def _repeat_pattern(values: list[str], size: int) -> list[str]:
    if size <= 0:
        return []
    repeats = (size + len(values) - 1) // len(values)
    return (values * repeats)[:size]


def _products_truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_id": [
                401,
                402,
                403,
                404,
                405,
                406,
                407,
                408,
                409,
                410,
                411,
                412,
                413,
                414,
                415,
                416,
                417,
                418,
            ],
            "product_name": [
                "Valve",
                "Sensor",
                "Pump",
                "Controller",
                "Actuator",
                "Regulator",
                "Flow Meter",
                "Gateway",
                "Relay",
                "Pressure Gauge",
                "Mixer",
                "Filter",
                "Turbine",
                "Bypass Kit",
                "Compressor",
                "PLC Module",
                "Thermal Probe",
                "Seal Kit",
            ],
            "unit_price": [
                15.0,
                48.0,
                73.0,
                105.0,
                64.0,
                39.0,
                57.0,
                122.0,
                33.0,
                46.0,
                88.0,
                27.0,
                133.0,
                22.0,
                97.0,
                149.0,
                35.0,
                19.0,
            ],
            "category": [
                "hardware",
                "iot",
                "hardware",
                "iot",
                "hardware",
                "ops",
                "iot",
                "platform",
                "ops",
                "hardware",
                "process",
                "ops",
                "industrial",
                "ops",
                "industrial",
                "platform",
                "iot",
                "hardware",
            ],
        }
    )


def _orders_truth(products_truth: pd.DataFrame) -> pd.DataFrame:
    batch_sizes = {
        "b1": 14,
        "b2": 16,
        "b3": 12,
        "b4": 20,
        "b5": 17,
        "b6": 15,
        "b7": 22,
        "b8": 13,
        "b9": 19,
        "b10": 16,
        "b11": 21,
        "b12": 14,
        "b13": 18,
    }
    product_ids = list(products_truth["product_id"])
    start_ts = pd.Timestamp("2026-03-28T06:00:00Z")

    records: list[dict[str, object]] = []
    order_id = 9001
    for batch_index, (batch_id, batch_size) in enumerate(batch_sizes.items()):
        for offset in range(batch_size):
            product_id = product_ids[(batch_index * 5 + offset * 3) % len(product_ids)]
            quantity = 1 + ((batch_index + 2 * offset) % 5)
            event_ts = (
                start_ts
                + pd.Timedelta(
                    minutes=12 * (order_id - 9001) + (batch_index % 3) * 4 + (offset % 4)
                )
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            records.append(
                {
                    "order_id": order_id,
                    "batch_id": batch_id,
                    "product_id": product_id,
                    "quantity": quantity,
                    "event_ts": event_ts,
                }
            )
            order_id += 1

    orders_truth = pd.DataFrame.from_records(records)
    orders_truth = orders_truth.merge(products_truth[["product_id", "unit_price"]], on="product_id", how="left")
    orders_truth["revenue"] = orders_truth["quantity"] * orders_truth["unit_price"]
    return orders_truth.drop(columns=["unit_price"])


def _summary_truth(orders_truth: pd.DataFrame) -> pd.DataFrame:
    return (
        orders_truth.assign(event_date=pd.to_datetime(orders_truth["event_ts"]).dt.strftime("%Y-%m-%d"))
        .groupby("event_date", as_index=False)
        .agg(order_count=("order_id", "count"), total_revenue=("revenue", "sum"))
    )


def _profile_manifest(profile: str) -> dict[str, object]:
    manifests: dict[str, dict[str, object]] = {
        "late_batch_resource_incident": {
            "profile_family": "familiar_incremental",
            "novelty_axes": ["resource_pressure", "late_batch", "retry_pressure"],
        },
        "schema_alias_unit_regression": {
            "profile_family": "familiar_incremental",
            "novelty_axes": ["schema_alias", "unit_drift", "summary_refresh"],
        },
        "stale_summary_oncall_recovery": {
            "profile_family": "familiar_incremental",
            "novelty_axes": ["stale_summary", "unsafe_commit_risk", "freshness_pressure"],
        },
        "heldout_task4_batch_shape_family": {
            "profile_family": "heldout_incremental",
            "novelty_axes": ["replay_window_shape_shift", "worker_pressure", "unsafe_commit_risk"],
        },
        "heldout_task4_summary_contract_family": {
            "profile_family": "heldout_incremental",
            "novelty_axes": ["summary_contract_drift", "unknown_aliases", "stale_summary"],
        },
        "heldout_task4_worker_pressure_family": {
            "profile_family": "heldout_incremental",
            "novelty_axes": ["worker_pressure", "backlog_burst", "freshness_pressure"],
        },
        "heldout_task4_freshness_pressure_family": {
            "profile_family": "heldout_incremental",
            "novelty_axes": ["freshness_pressure", "mixed_stale_downstreams", "summary_contract_drift"],
        },
    }
    return manifests[profile]


def _mutate_orders_for_profile(profile: str, visible_orders: pd.DataFrame, pending_orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    visible = visible_orders.copy()
    pending = pending_orders.copy()
    if profile == "late_batch_resource_incident":
        visible["product_id"] = visible["product_id"].astype(str)
        visible["quantity"] = _repeat_pattern(["2 units", "1", "3", "2", "5 units", "2", "1", "4"], len(visible))
        visible["event_ts"] = _repeat_pattern([
            "28/03/2026 10:00",
            "2026-03-28T10:05:00Z",
            "03-28-2026 10:15",
            "2026-03-28T10:35:00Z",
            "03-29-2026 09:10",
            "2026-03-29T09:40:00Z",
            "03-29-2026 10:00",
            "2026-03-29T10:30:00Z",
        ], len(visible))
        return visible, pending
    if profile == "schema_alias_unit_regression":
        visible["product_id"] = visible["product_id"].astype(str)
        visible["quantity"] = _repeat_pattern(["2", "1 units", "3", "2", "5", "2 units", "1", "4"], len(visible))
        visible = visible.rename(columns={"event_ts": "event_time"})
        pending = pending.rename(columns={"event_ts": "event_time"})
        visible["event_time"] = _repeat_pattern([
            "28/03/2026 10:00",
            "2026-03-28T10:05:00Z",
            "03-28-2026 10:15",
            "2026-03-28T10:35:00Z",
            "2026-03-29T09:10:00Z",
            "03-29-2026 09:40",
            "2026-03-29T10:00:00Z",
            "2026-03-29T10:30:00Z",
        ], len(visible))
        return visible, pending
    if profile == "stale_summary_oncall_recovery":
        visible["product_id"] = visible["product_id"].astype(str)
        visible["quantity"] = _repeat_pattern(["2", "1", "3 units", "2", "5 units", "2", "1", "4"], len(visible))
        visible["event_ts"] = _repeat_pattern([
            "2026-03-28T10:00:00Z",
            "2026-03-28T10:05:00Z",
            "03-28-2026 10:15",
            "2026-03-28T10:35:00Z",
            "03-29-2026 09:10",
            "2026-03-29T09:40:00Z",
            "2026-03-29T10:00:00Z",
            "03-29-2026 10:30",
        ], len(visible))
        return visible, pending
    if profile == "heldout_task4_batch_shape_family":
        visible["product_id"] = visible["product_id"].astype(str)
        visible["quantity"] = _repeat_pattern(["2 units", "1", "3", "2", "5 units", "2", "1", "4"], len(visible))
        visible = visible.rename(columns={"event_ts": "load_window_start"})
        pending = pending.rename(columns={"event_ts": "load_window_start"})
        visible["load_window_start"] = _repeat_pattern([
            "2026-03-28 15:30:00+05:30",
            "2026-03-28T10:05:00Z",
            "28/03/2026 10:15",
            "2026-03-28T10:35:00Z",
            "03-29-2026 09:10",
            "2026-03-29T09:40:00Z",
            "29/03/2026 10:00",
            "2026-03-29T10:30:00Z",
        ], len(visible))
        return visible, pending
    if profile == "heldout_task4_summary_contract_family":
        visible["product_id"] = visible["product_id"].astype(str)
        visible = visible.rename(columns={"event_ts": "processing_ts", "batch_id": "window_id"})
        pending = pending.rename(columns={"event_ts": "processing_ts", "batch_id": "window_id"})
        visible["processing_ts"] = _repeat_pattern([
            "2026-03-28 15:30:00+05:30",
            "2026-03-28T10:05:00Z",
            "28/03/2026 10:15",
            "2026-03-28T10:35:00Z",
            "2026-03-29T09:10:00Z",
            "03-29-2026 09:40",
            "29/03/2026 10:00",
            "2026-03-29T10:30:00Z",
        ], len(visible))
        visible["quantity"] = _repeat_pattern(["2 units", "1", "3", "2", "5 units", "2", "1", "4"], len(visible))
        return visible, pending
    if profile == "heldout_task4_worker_pressure_family":
        visible["product_id"] = visible["product_id"].astype(str)
        visible["quantity"] = _repeat_pattern(["2 units", "1", "3", "2", "5 units", "2", "1 units", "4"], len(visible))
        visible = visible.rename(columns={"event_ts": "event_time"})
        pending = pending.rename(columns={"event_ts": "event_time"})
        visible["event_time"] = _repeat_pattern([
            "2026-03-28 15:30:00+05:30",
            "2026-03-28T10:05:00Z",
            "28/03/2026 10:15",
            "2026-03-28T10:35:00Z",
            "03-29-2026 09:10",
            "2026-03-29T09:40:00Z",
            "29/03/2026 10:00",
            "2026-03-29T10:30:00Z",
        ], len(visible))
        return visible, pending
    visible["product_id"] = visible["product_id"].astype(str)
    visible["quantity"] = _repeat_pattern(["2 units", "1", "3", "2", "5 units", "2", "1", "4"], len(visible))
    visible = visible.rename(columns={"event_ts": "window_ts", "batch_id": "replay_window_id"})
    pending = pending.rename(columns={"event_ts": "window_ts", "batch_id": "replay_window_id"})
    visible["window_ts"] = _repeat_pattern([
        "2026-03-28 15:30:00+05:30",
        "2026-03-28T10:05:00Z",
        "28/03/2026 10:15",
        "2026-03-28T10:35:00Z",
        "03-29-2026 09:10",
        "2026-03-29T09:40:00Z",
        "29/03/2026 10:00",
        "2026-03-29T10:30:00Z",
    ], len(visible))
    return visible, pending


def _mutate_products_for_profile(profile: str, products_truth: pd.DataFrame) -> pd.DataFrame:
    products = products_truth.copy()
    if profile in {"late_batch_resource_incident", "stale_summary_oncall_recovery"}:
        products["unit_price"] = _repeat_pattern(
            ["$15.00", "$48.00", "73.0", "$105.00", "6400 cents", "$39.00"],
            len(products),
        )
        return products
    if profile == "schema_alias_unit_regression":
        products["unit_price"] = _repeat_pattern(
            ["$15.00", "4800 cents", "$73.00", "10500 cents", "$64.00", "3900 cents"],
            len(products),
        )
        products = products.rename(columns={"category": "product_segment"})
        products.loc[1, "product_segment"] = " IoT "
        return products
    if profile == "heldout_task4_summary_contract_family":
        products["unit_price"] = _repeat_pattern(
            ["$15.00", "4800 cents", "$73.00", "10500 cents", "$64.00", "3900 cents"],
            len(products),
        )
        products = products.rename(columns={"product_id": "sku_key", "category": "product_family"})
        products["sku_key"] = products["sku_key"].astype(str)
        products.loc[1, "product_family"] = " IoT "
        return products
    if profile == "heldout_task4_freshness_pressure_family":
        products["unit_price"] = _repeat_pattern(
            ["$15.00", "4800 cents", "$73.00", "$105.00", "$64.00", "3900 cents"],
            len(products),
        )
        products = products.rename(columns={"category": "segment_name"})
        products.loc[1, "segment_name"] = " IoT "
        return products
    products["unit_price"] = _repeat_pattern(
        ["$15.00", "4800 cents", "$73.00", "$105.00", "$64.00", "3900 cents"],
        len(products),
    )
    return products


def _mutate_summary_for_profile(profile: str, summary_truth: pd.DataFrame) -> pd.DataFrame:
    summary = summary_truth[summary_truth["event_date"] != "2026-03-30"].copy()
    if profile in {"late_batch_resource_incident", "schema_alias_unit_regression"}:
        summary["total_revenue"] = (summary["total_revenue"] * 0.91).round(2)
        return summary
    if profile == "stale_summary_oncall_recovery":
        summary = summary.rename(columns={"event_date": "business_date"})
        summary["total_revenue"] = (summary["total_revenue"] * 0.91).round(2)
        return summary
    if profile == "heldout_task4_batch_shape_family":
        summary["total_revenue"] = (summary["total_revenue"] * 0.89).round(2)
        summary = summary.rename(columns={"event_date": "summary_day"})
        return summary
    if profile == "heldout_task4_summary_contract_family":
        summary = summary.rename(columns={"event_date": "summary_day", "total_revenue": "booked_revenue_value"})
        summary["booked_revenue_value"] = (summary["booked_revenue_value"] * 0.87).round(2)
        return summary
    if profile == "heldout_task4_worker_pressure_family":
        summary = summary.rename(columns={"event_date": "business_date"})
        summary["total_revenue"] = (summary["total_revenue"] * 0.9).round(2)
        return summary
    summary = summary.rename(columns={"event_date": "recovery_day", "total_revenue": "gross_sales_value"})
    summary["gross_sales_value"] = (summary["gross_sales_value"] * 0.86).round(2)
    return summary


def load_task4_fixture(profile: str, split: str) -> IncidentFixture:
    products_truth = _products_truth()
    orders_truth = _orders_truth(products_truth)
    summary_truth = _summary_truth(orders_truth)
    profile_manifest = _profile_manifest(profile)
    visible_batch_ids = ["b1", "b2"] if profile_manifest["profile_family"] == "heldout_incremental" else ["b1", "b2", "b3"]
    visible_orders = orders_truth[orders_truth["batch_id"].isin(visible_batch_ids)].copy()
    pending_orders = orders_truth[~orders_truth["batch_id"].isin(visible_batch_ids)].copy()
    orders_broken, pending_orders = _mutate_orders_for_profile(profile, visible_orders, pending_orders)
    products_broken = _mutate_products_for_profile(profile, products_truth)
    summary_broken = _mutate_summary_for_profile(profile, summary_truth)

    backlog_rows = len(pending_orders)
    pending_batch_column = "batch_id" if "batch_id" in pending_orders.columns else next(
        column for column in pending_orders.columns if column.endswith("window_id") or column == "window_id"
    )
    pending_batches = int(pending_orders[pending_batch_column].nunique()) if backlog_rows > 0 else 0
    replay_window_row_counts = {
        str(batch_id): int(count)
        for batch_id, count in orders_truth["batch_id"].astype(str).value_counts().sort_index().items()
    }
    visible_replay_window_row_counts = {
        str(batch_id): int(count)
        for batch_id, count in visible_orders["batch_id"].astype(str).value_counts().sort_index().items()
    }
    pending_replay_window_row_counts = {
        str(batch_id): int(count)
        for batch_id, count in pending_orders[pending_batch_column].astype(str).value_counts().sort_index().items()
    }
    watermark_before = pd.to_datetime(visible_orders["event_ts"], utc=True).max().strftime("%Y-%m-%dT%H:%M:%SZ")
    expected_watermark_after_replay = pd.to_datetime(orders_truth["event_ts"], utc=True).max().strftime("%Y-%m-%dT%H:%M:%SZ")
    required_resource_level = 3 if pending_batches >= 3 or profile_manifest["profile_family"] == "heldout_incremental" else 2
    freshness_lag_minutes = 120 if split == "train" else 180
    backlog_age_minutes = 180 if split == "train" else 300
    if profile_manifest["profile_family"] == "heldout_incremental":
        freshness_lag_minutes += 45
        backlog_age_minutes += 60
    incident_id = f"t4-{profile}-{split}"
    metadata = {
        "pending_orders": pending_orders,
        "backlog_rows": backlog_rows,
        "queue_backlog_age_minutes": backlog_age_minutes,
        "freshness_lag_minutes": freshness_lag_minutes,
        "resource_level": 1,
        "required_resource_level": required_resource_level,
        "pending_batches": pending_batches,
        "downstream_stale": True,
        "workload_pressure": 0.82 if split == "train" else 0.94,
        "incident_manifest": {
            "incident_id": incident_id,
            "dag_id": "incremental_orders_refresh",
            "warehouse": "ops_wh",
            "severity": "critical" if profile_manifest["profile_family"] == "heldout_incremental" else ("critical" if split == "eval" else "high"),
            "failed_tasks": ["load_incremental_orders", "refresh_daily_summary"],
            "downstream_assets": ["daily_summary", "sales_dashboard", "ops_watermark_audit"],
            "profile_family": profile_manifest["profile_family"],
            "novelty_axes": profile_manifest["novelty_axes"],
            "affected_batch_ids": sorted(pending_replay_window_row_counts),
            "replay_window_row_counts": replay_window_row_counts,
            "visible_replay_window_row_counts": visible_replay_window_row_counts,
            "pending_replay_window_row_counts": pending_replay_window_row_counts,
            "watermark_before": watermark_before,
            "expected_watermark_after_replay": expected_watermark_after_replay,
            "summary_contract_version": "v3-summary-contract" if profile_manifest["profile_family"] == "heldout_incremental" else "v2-summary-contract",
            "source_contract_version": "v3-replay-window-contract" if profile_manifest["profile_family"] == "heldout_incremental" else "v2-orders-stream",
        },
        "dag_runs": pd.DataFrame(
            [
                {"run_id": f"{incident_id}-1", "task_id": "extract_orders", "state": "success", "attempt": 1},
                {"run_id": f"{incident_id}-1", "task_id": "load_incremental_orders", "state": "failed", "attempt": 2},
                {"run_id": f"{incident_id}-1", "task_id": "refresh_daily_summary", "state": "failed", "attempt": 1},
                {"run_id": f"{incident_id}-1", "task_id": "publish_summary_assets", "state": "upstream_failed", "attempt": 0},
            ]
        ),
        "warehouse_events": pd.DataFrame(
            [
                {"ts": "2026-03-30T09:35:00Z", "system": "airflow", "event": "incremental_batch_retry_exhausted"},
                {"ts": "2026-03-30T09:37:00Z", "system": "warehouse", "event": "summary_table_stale"},
                {"ts": "2026-03-30T09:40:00Z", "system": "autoscaler", "event": "worker_pool_under_provisioned"},
                {"ts": "2026-03-30T09:44:00Z", "system": "freshness_monitor", "event": "watermark_lag_breach"},
            ]
        ),
        "trace_drift_markers": [
            "incremental_replay_backlog",
            "summary_refresh_staleness",
            "worker_pool_pressure",
            *profile_manifest["novelty_axes"],
        ],
        "trace_dependency_health": {
            "incremental_backlog": "pending_replay",
            "summary_state": "stale",
            "recovery_gate": "recovery_incomplete",
            "replay_window_state": "irregular" if profile_manifest["profile_family"] == "heldout_incremental" else "expected",
        },
        "recent_failure_counters": {
            "incremental_load_failures": 2 if profile_manifest["profile_family"] == "heldout_incremental" else 1,
            "summary_refresh_failures": 1,
            "resource_scale_attempts": 1 if required_resource_level > 1 else 0,
        },
        "operational_trace_summary": "Bundled airflow-style task runs, watermark lag alerts, and replay-window manifests describe an incremental ETL recovery incident.",
    }
    return IncidentFixture(
        broken_tables={
            "orders": orders_broken,
            "products": products_broken,
            "daily_summary": summary_broken,
        },
        ground_truth_tables={
            "orders": orders_truth,
            "products": products_truth,
            "daily_summary": summary_truth,
        },
        metadata=metadata,
    )
