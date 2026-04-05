from __future__ import annotations

import pandas as pd

from .shared import IncidentFixture


def _repeat_pattern(values: list[str], size: int) -> list[str]:
    if size <= 0:
        return []
    repeats = (size + len(values) - 1) // len(values)
    return (values * repeats)[:size]


def _catalog_truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_id": [
                601,
                602,
                603,
                604,
                605,
                606,
                607,
                608,
                609,
                610,
                611,
                612,
                613,
                614,
                615,
                616,
                617,
                618,
            ],
            "product_name": [
                "Valve",
                "Sensor",
                "Pump",
                "Controller",
                "Relay",
                "Actuator",
                "Flow Meter",
                "Gateway",
                "Pressure Gauge",
                "Regulator",
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
                18.0,
                52.0,
                77.0,
                112.0,
                41.0,
                68.0,
                57.0,
                121.0,
                46.0,
                39.0,
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
                "ops",
                "industrial",
                "iot",
                "platform",
                "hardware",
                "ops",
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


def _source_truth(catalog_truth: pd.DataFrame) -> pd.DataFrame:
    batch_sizes = {
        "t1": 12,
        "t2": 11,
        "t3": 15,
        "t4": 9,
        "t5": 14,
        "t6": 13,
        "t7": 16,
        "t8": 10,
        "t9": 18,
        "t10": 12,
        "t11": 15,
        "t12": 11,
    }
    product_ids = list(catalog_truth["product_id"])
    start_ts = pd.Timestamp("2026-03-28T22:00:00Z")

    records: list[dict[str, object]] = []
    order_id = 11001
    event_index = 0

    for batch_index, (batch_id, batch_size) in enumerate(batch_sizes.items()):
        for row_offset in range(batch_size):
            product_id = product_ids[(batch_index * 5 + row_offset * 3) % len(product_ids)]
            quantity = 1 + ((batch_index + 2 * row_offset) % 5)
            event_ts = (
                start_ts
                + pd.Timedelta(
                    minutes=(12 * event_index) + (batch_index % 3) * 5 + (row_offset % 4)
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
            event_index += 1

    source_truth = pd.DataFrame.from_records(records)
    source_truth = source_truth.merge(
        catalog_truth[["product_id", "unit_price"]],
        on="product_id",
        how="left",
    )
    source_truth["gross_revenue"] = source_truth["quantity"] * source_truth["unit_price"]
    return source_truth.drop(columns=["unit_price"])


def _rollup_truth(source_truth: pd.DataFrame) -> pd.DataFrame:
    return (
        source_truth.assign(
            hour_bucket=pd.to_datetime(source_truth["event_ts"]).dt.strftime(
                "%Y-%m-%dT%H:00:00Z"
            )
        )
        .groupby("hour_bucket", as_index=False)
        .agg(order_count=("order_id", "count"), gross_revenue=("gross_revenue", "sum"))
    )


def _profile_manifest(profile: str) -> dict[str, object]:
    manifests: dict[str, dict[str, object]] = {
        "temporal_rollup_backfill_incident": {
            "profile_family": "familiar_temporal",
            "novelty_axes": ["late_correction_replay", "rollup_backfill_pressure"],
            "failed_tasks": ["replay_late_batches", "refresh_hourly_rollup"],
            "source_contract_version": "v2-source-orders",
            "rollup_contract_version": "v2-hourly-rollup",
            "downstream_assets": ["hourly_rollup", "revenue_monitor"],
        },
        "schema_evolution_backfill_recovery": {
            "profile_family": "familiar_temporal",
            "novelty_axes": ["schema_alias_shift", "catalog_contract_drift"],
            "failed_tasks": ["sync_catalog", "replay_late_batches", "refresh_hourly_rollup"],
            "source_contract_version": "v3-source-alias",
            "rollup_contract_version": "v2-hourly-rollup",
            "downstream_assets": ["hourly_rollup", "revenue_monitor"],
        },
        "late_correction_backpressure_incident": {
            "profile_family": "familiar_temporal",
            "novelty_axes": ["late_correction_replay", "rollup_window_alias"],
            "failed_tasks": ["replay_late_batches", "refresh_hourly_rollup"],
            "source_contract_version": "v2-source-orders",
            "rollup_contract_version": "v2-window-start-alias",
            "downstream_assets": ["hourly_rollup", "revenue_monitor"],
        },
        "heldout_temporal_schema_extension_family": {
            "profile_family": "heldout_temporal",
            "novelty_axes": [
                "unknown_source_aliases",
                "schema_extension",
                "catalog_contract_drift",
            ],
            "failed_tasks": [
                "load_source_orders",
                "sync_catalog",
                "replay_late_batches",
                "refresh_hourly_rollup",
            ],
            "source_contract_version": "v4-source-schema-extension",
            "rollup_contract_version": "v3-window-start-gross-sales",
            "downstream_assets": ["hourly_rollup", "revenue_monitor", "ops_watermark_audit"],
        },
        "heldout_temporal_rollup_contract_family": {
            "profile_family": "heldout_temporal",
            "novelty_axes": [
                "unknown_catalog_aliases",
                "rollup_contract_drift",
                "backfill_pressure",
            ],
            "failed_tasks": ["sync_catalog", "replay_late_batches", "refresh_hourly_rollup"],
            "source_contract_version": "v3-observed-at",
            "rollup_contract_version": "v4-rollup-revenue-total",
            "downstream_assets": ["hourly_rollup", "revenue_monitor", "margin_dashboard"],
        },
        "heldout_temporal_correction_replay_family": {
            "profile_family": "heldout_temporal",
            "novelty_axes": [
                "unknown_source_aliases",
                "late_correction_replay",
                "catalog_contract_drift",
            ],
            "failed_tasks": ["load_source_orders", "replay_late_batches", "refresh_hourly_rollup"],
            "source_contract_version": "v4-source-sku-time",
            "rollup_contract_version": "v3-window-start-gross-sales",
            "downstream_assets": [
                "hourly_rollup",
                "revenue_monitor",
                "correction_audit_feed",
            ],
        },
    }
    return manifests[profile]


def load_task5_fixture(profile: str, split: str) -> IncidentFixture:
    catalog_truth = _catalog_truth()
    source_truth = _source_truth(catalog_truth)
    rollup_truth = _rollup_truth(source_truth)
    profile_manifest = _profile_manifest(profile)

    visible_batch_ids = ["t1", "t2", "t3"]
    if profile_manifest["profile_family"] == "familiar_temporal":
        visible_batch_ids.append("t4")

    visible_orders = source_truth[source_truth["batch_id"].isin(visible_batch_ids)].copy()
    pending_orders = source_truth[~source_truth["batch_id"].isin(visible_batch_ids)].copy()
    pending_orders_for_metrics = pending_orders.copy()

    source_broken = visible_orders.copy()
    catalog_broken = catalog_truth.copy()

    rollup_cutoff = (
        pd.to_datetime(visible_orders["event_ts"], utc=True)
        .dt.floor("h")
        .max()
        .strftime("%Y-%m-%dT%H:00:00Z")
    )
    rollup_broken = rollup_truth[rollup_truth["hour_bucket"] <= rollup_cutoff].copy()

    if profile == "temporal_rollup_backfill_incident":
        source_broken["product_id"] = source_broken["product_id"].astype(str)
        source_broken["quantity"] = _repeat_pattern(
            ["2 units", "1", "3 units", "2", "4", "1 units", "5", "2 units"],
            len(source_broken),
        )
        source_broken["event_ts"] = _repeat_pattern(
            [
                "29/03/2026 00:40",
                "2026-03-29T00:53:00Z",
                "2026-03-29 01:05:00+05:30",
                "29/03/2026 01:19",
                "2026-03-29T01:31:00Z",
                "2026-03-29 01:47:00+05:30",
                "2026-03-29T02:03:00Z",
                "29/03/2026 02:22",
            ],
            len(source_broken),
        )
        source_broken["gross_revenue"] = (visible_orders["gross_revenue"] * 0.83).round(2).astype(object)
        catalog_broken["unit_price"] = _repeat_pattern(
            ["$18.00", "$52.00", "7700 cents", "$112.00", "$41.00", "6800 cents"],
            len(catalog_broken),
        )
        rollup_broken.loc[:, "gross_revenue"] = (rollup_broken["gross_revenue"] * 0.88).round(2)
        if not rollup_broken.empty:
            rollup_broken.loc[rollup_broken.index[0], "hour_bucket"] = "29/03/2026 00:00"

    elif profile == "schema_evolution_backfill_recovery":
        source_broken["product_id"] = source_broken["product_id"].astype(str)
        source_broken["quantity"] = _repeat_pattern(
            ["2 units", "1", "3 units", "2", "missing", "1 units", "4", "2"],
            len(source_broken),
        )
        source_broken = source_broken.rename(columns={"event_ts": "observed_at"})
        source_broken["observed_at"] = _repeat_pattern(
            [
                "29/03/2026 00:40",
                "2026-03-29T00:53:00Z",
                "2026-03-29 01:05:00+05:30",
                "29/03/2026 01:19",
                "2026-03-29T01:31:00Z",
                "2026-03-29 01:47:00+05:30",
                "2026-03-29T02:03:00Z",
                "29/03/2026 02:22",
            ],
            len(source_broken),
        )
        source_broken["gross_revenue"] = (visible_orders["gross_revenue"] * 0.83).round(2).astype(object)
        catalog_broken["unit_price"] = _repeat_pattern(
            ["$18.00", "5200 cents", "7700 cents", "$112.00", "$41.00", "6800 cents"],
            len(catalog_broken),
        )
        catalog_broken = catalog_broken.rename(columns={"category": "product_segment"})
        catalog_broken.loc[1, "product_segment"] = " IoT "
        catalog_broken = pd.concat([catalog_broken, catalog_broken.iloc[[2]]], ignore_index=True)
        rollup_broken.loc[:, "gross_revenue"] = (rollup_broken["gross_revenue"] * 0.88).round(2)
        if not rollup_broken.empty:
            rollup_broken.loc[rollup_broken.index[0], "hour_bucket"] = "29/03/2026 00:00"

    elif profile == "late_correction_backpressure_incident":
        source_broken["product_id"] = source_broken["product_id"].astype(str)
        source_broken["quantity"] = _repeat_pattern(
            ["2", "1 units", "3", "2 units", "4", "missing", "5", "2"],
            len(source_broken),
        )
        source_broken["event_ts"] = _repeat_pattern(
            [
                "29/03/2026 00:40",
                "2026-03-29T00:53:00Z",
                "2026-03-29 01:05:00+05:30",
                "29/03/2026 01:19",
                "2026-03-29T01:31:00Z",
                "2026-03-29 01:47:00+05:30",
                "2026-03-29T02:03:00Z",
                "29/03/2026 02:22",
            ],
            len(source_broken),
        )
        source_broken["gross_revenue"] = (visible_orders["gross_revenue"] * 0.83).round(2).astype(object)
        catalog_broken["unit_price"] = _repeat_pattern(
            ["$18.00", "$52.00", "7700 cents", "$112.00", "$41.00", "6800 cents"],
            len(catalog_broken),
        )
        catalog_broken.loc[1, "category"] = " IoT "
        catalog_broken = pd.concat([catalog_broken, catalog_broken.iloc[[2]]], ignore_index=True)
        rollup_broken.loc[:, "gross_revenue"] = (rollup_broken["gross_revenue"] * 0.88).round(2)
        rollup_broken = rollup_broken.rename(columns={"hour_bucket": "window_start"})
        if not rollup_broken.empty:
            rollup_broken.loc[rollup_broken.index[0], "window_start"] = "29/03/2026 00:00"

    elif profile == "heldout_temporal_schema_extension_family":
        source_broken["product_id"] = source_broken["product_id"].astype(str)
        source_broken["quantity"] = _repeat_pattern(
            ["2 units", "missing", "3 units", "2", "4", "1 units", "5", "2"],
            len(source_broken),
        )
        pending_orders = pending_orders.rename(
            columns={
                "event_ts": "source_snapshot_utc",
                "product_id": "sku_key",
                "quantity": "ordered_units",
            }
        )
        pending_orders["sku_key"] = pending_orders["sku_key"].astype(str)
        source_broken = source_broken.rename(
            columns={
                "event_ts": "source_snapshot_utc",
                "product_id": "sku_key",
                "quantity": "ordered_units",
            }
        )
        source_broken["source_snapshot_utc"] = _repeat_pattern(
            [
                "29/03/2026 00:40",
                "2026-03-29T00:53:00Z",
                "2026-03-29 01:05:00+05:30",
                "29/03/2026 01:19",
                "2026-03-29T01:31:00Z",
                "2026-03-29 01:47:00+05:30",
                "2026-03-29T02:03:00Z",
                "29/03/2026 02:22",
            ],
            len(source_broken),
        )
        source_broken["recognized_revenue_value"] = (
            visible_orders["gross_revenue"] * 0.77
        ).round(2).astype(object)
        source_broken = source_broken.drop(columns=["gross_revenue"])
        catalog_broken["unit_price"] = _repeat_pattern(
            ["$18.00", "5200 cents", "7700 cents", "$112.00", "$41.00", "6800 cents"],
            len(catalog_broken),
        )
        catalog_broken = catalog_broken.rename(columns={"category": "catalog_family_group"})
        catalog_broken.loc[1, "catalog_family_group"] = " IoT "
        catalog_broken = pd.concat([catalog_broken, catalog_broken.iloc[[2]]], ignore_index=True)
        rollup_broken = rollup_broken.rename(
            columns={
                "hour_bucket": "rollup_bucket_start_utc",
                "gross_revenue": "recognized_revenue_value",
            }
        )
        rollup_broken.loc[:, "recognized_revenue_value"] = (
            rollup_broken["recognized_revenue_value"] * 0.83
        ).round(2)
        if not rollup_broken.empty:
            rollup_broken.loc[
                rollup_broken.index[0],
                "rollup_bucket_start_utc",
            ] = "29/03/2026 00:00"

    elif profile == "heldout_temporal_rollup_contract_family":
        source_broken["product_id"] = source_broken["product_id"].astype(str)
        source_broken["quantity"] = _repeat_pattern(
            ["2 units", "1", "3 units", "2", "missing", "1 units", "4", "2"],
            len(source_broken),
        )
        pending_orders = pending_orders.rename(
            columns={
                "event_ts": "observed_watermark_at",
                "product_id": "inventory_sku_key",
            }
        )
        pending_orders["inventory_sku_key"] = pending_orders["inventory_sku_key"].astype(str)
        source_broken = source_broken.rename(
            columns={
                "event_ts": "observed_watermark_at",
                "product_id": "inventory_sku_key",
            }
        )
        source_broken["observed_watermark_at"] = _repeat_pattern(
            [
                "29/03/2026 00:40",
                "2026-03-29T00:53:00Z",
                "2026-03-29 01:05:00+05:30",
                "29/03/2026 01:19",
                "2026-03-29T01:31:00Z",
                "2026-03-29 01:47:00+05:30",
                "2026-03-29T02:03:00Z",
                "29/03/2026 02:22",
            ],
            len(source_broken),
        )
        source_broken["booked_revenue_usd"] = (visible_orders["gross_revenue"] * 0.79).round(2).astype(object)
        source_broken = source_broken.drop(columns=["gross_revenue"])
        catalog_broken["unit_price"] = _repeat_pattern(
            ["$18.00", "$52.00", "7700 cents", "$112.00", "$41.00", "6800 cents"],
            len(catalog_broken),
        )
        catalog_broken = catalog_broken.rename(columns={"unit_price": "unit_cost_cents"})
        rollup_broken = rollup_broken.rename(
            columns={"hour_bucket": "rollup_window_start_utc", "gross_revenue": "net_sales_value"}
        )
        rollup_broken.loc[:, "net_sales_value"] = (rollup_broken["net_sales_value"] * 0.84).round(2)
        if not rollup_broken.empty:
            rollup_broken.loc[
                rollup_broken.index[0],
                "rollup_window_start_utc",
            ] = "29/03/2026 00:00"

    elif profile == "heldout_temporal_correction_replay_family":
        source_broken["product_id"] = source_broken["product_id"].astype(str)
        source_broken["quantity"] = _repeat_pattern(
            ["2 units", "1", "3 units", "2", "missing", "1 units", "4", "2"],
            len(source_broken),
        )
        pending_orders = pending_orders.rename(columns={"event_ts": "replay_observed_at", "product_id": "item_sku"})
        pending_orders["item_sku"] = pending_orders["item_sku"].astype(str)
        source_broken = source_broken.rename(columns={"event_ts": "replay_observed_at", "product_id": "item_sku"})
        source_broken["replay_observed_at"] = _repeat_pattern(
            [
                "29/03/2026 00:40",
                "2026-03-29T00:53:00Z",
                "2026-03-29 01:05:00+05:30",
                "29/03/2026 01:19",
                "2026-03-29T01:31:00Z",
                "2026-03-29 01:47:00+05:30",
                "2026-03-29T02:03:00Z",
                "29/03/2026 02:22",
            ],
            len(source_broken),
        )
        source_broken["revenue_usd_value"] = (visible_orders["gross_revenue"] * 0.76).round(2).astype(object)
        source_broken = source_broken.drop(columns=["gross_revenue"])
        catalog_broken["unit_price"] = _repeat_pattern(
            ["$18.00", "5200 cents", "7700 cents", "$112.00", "$41.00", "6800 cents"],
            len(catalog_broken),
        )
        catalog_broken = catalog_broken.rename(columns={"category": "catalog_segment_name"})
        catalog_broken.loc[1, "catalog_segment_name"] = " IoT "
        catalog_broken = pd.concat([catalog_broken, catalog_broken.iloc[[2]]], ignore_index=True)
        rollup_broken = rollup_broken.rename(
            columns={"hour_bucket": "bucket_window_utc_start", "gross_revenue": "replayed_revenue_value"}
        )
        rollup_broken.loc[:, "replayed_revenue_value"] = (
            rollup_broken["replayed_revenue_value"] * 0.82
        ).round(2)
        if not rollup_broken.empty:
            rollup_broken.loc[
                rollup_broken.index[0],
                "bucket_window_utc_start",
            ] = "29/03/2026 00:00"

    backlog_rows = len(pending_orders)
    pending_batches = int(pending_orders["batch_id"].nunique()) if backlog_rows > 0 else 0
    required_resource_level = (
        3
        if pending_batches >= 5
        or backlog_rows >= 80
        or profile_manifest["profile_family"] == "heldout_temporal"
        else 2
    )

    freshness_lag_minutes = 180 if split == "eval" else 120
    backlog_age_minutes = 360 if split == "train" else 480
    if profile_manifest["profile_family"] == "heldout_temporal":
        freshness_lag_minutes += 45
        backlog_age_minutes += 90

    watermark_before = pd.to_datetime(visible_orders["event_ts"], utc=True).max().strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    expected_watermark_after_replay = pd.to_datetime(source_truth["event_ts"], utc=True).max().strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    affected_hour_buckets = sorted(
        pd.to_datetime(pending_orders_for_metrics["event_ts"], utc=True)
        .dt.floor("h")
        .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        .unique()
        .tolist()
    )
    late_correction_order_ids = (
        pending_orders_for_metrics["order_id"].astype(int).head(12).tolist()
    )
    replay_window_row_counts = {
        str(batch_id): int(count)
        for batch_id, count in source_truth["batch_id"].astype(str).value_counts().sort_index().items()
    }
    visible_replay_window_row_counts = {
        str(batch_id): int(count)
        for batch_id, count in visible_orders["batch_id"].astype(str).value_counts().sort_index().items()
    }
    pending_replay_window_row_counts = {
        str(batch_id): int(count)
        for batch_id, count in pending_orders["batch_id"].astype(str).value_counts().sort_index().items()
    }

    incident_id = f"t5-{profile}-{split}"
    workload_pressure = 0.78 if split == "train" else 0.93
    if profile_manifest["profile_family"] == "heldout_temporal":
        workload_pressure = min(workload_pressure + 0.03, 0.98)

    metadata = {
        "pending_orders": pending_orders,
        "backlog_rows": backlog_rows,
        "queue_backlog_age_minutes": backlog_age_minutes,
        "freshness_lag_minutes": freshness_lag_minutes,
        "resource_level": 1,
        "required_resource_level": required_resource_level,
        "pending_batches": pending_batches,
        "downstream_stale": True,
        "workload_pressure": workload_pressure,
        "incident_manifest": {
            "incident_id": incident_id,
            "dag_id": "temporal_orders_rollup",
            "warehouse": "finance_wh",
            "severity": "critical" if split == "eval" else "high",
            "failed_tasks": profile_manifest["failed_tasks"],
            "downstream_assets": profile_manifest["downstream_assets"],
            "profile_family": profile_manifest["profile_family"],
            "novelty_axes": profile_manifest["novelty_axes"],
            "affected_hour_buckets": affected_hour_buckets,
            "late_correction_order_ids": late_correction_order_ids,
            "watermark_before": watermark_before,
            "expected_watermark_after_replay": expected_watermark_after_replay,
            "source_contract_version": profile_manifest["source_contract_version"],
            "rollup_contract_version": profile_manifest["rollup_contract_version"],
            "replay_window_row_counts": replay_window_row_counts,
            "visible_replay_window_row_counts": visible_replay_window_row_counts,
            "pending_replay_window_row_counts": pending_replay_window_row_counts,
        },
        "dag_runs": pd.DataFrame(
            [
                {
                    "run_id": f"{incident_id}-1",
                    "task_id": "load_source_orders",
                    "state": "failed" if "load_source_orders" in profile_manifest["failed_tasks"] else "success",
                    "attempt": 2,
                },
                {
                    "run_id": f"{incident_id}-1",
                    "task_id": "sync_catalog",
                    "state": "failed" if "sync_catalog" in profile_manifest["failed_tasks"] else "success",
                    "attempt": 2,
                },
                {
                    "run_id": f"{incident_id}-1",
                    "task_id": "replay_late_batches",
                    "state": "failed",
                    "attempt": 1,
                },
                {
                    "run_id": f"{incident_id}-1",
                    "task_id": "refresh_hourly_rollup",
                    "state": "failed",
                    "attempt": 1,
                },
            ]
        ),
        "warehouse_events": pd.DataFrame(
            [
                {
                    "ts": "2026-03-29T15:40:00Z",
                    "system": "airflow",
                    "event": "late_batch_replay_blocked",
                },
                {
                    "ts": "2026-03-29T15:42:00Z",
                    "system": "warehouse",
                    "event": "hourly_rollup_stale",
                },
                {
                    "ts": "2026-03-29T15:45:00Z",
                    "system": "schema_registry",
                    "event": "temporal_contract_shift_detected",
                },
                {
                    "ts": "2026-03-29T15:47:00Z",
                    "system": "watermark_monitor",
                    "event": f"watermark_stalled_at:{watermark_before}",
                },
            ]
        ),
        "trace_drift_markers": [
            "temporal_contract_shift",
            "late_batch_replay_backpressure",
            "hourly_rollup_consistency_alarm",
            "replay_window_shape_shift",
        ],
        "trace_dependency_health": {
            "schema_alignment": "schema_drift_active",
            "temporal_backfill": "late_batches_pending",
            "rollup_state": "rollup_stale",
            "temporal_closure": "replay_window_open",
        },
        "recent_failure_counters": {
            "late_correction_failures": 3,
            "rollup_refresh_failures": 2,
            "schema_migration_regressions": 1,
        },
        "operational_trace_summary": (
            "Bundled DAG-task and warehouse traces capture replay windows, stalled watermarks, "
            "contract-version drift, and stale temporal rollups at higher incident scale."
        ),
    }

    if profile_manifest["profile_family"] == "heldout_temporal":
        metadata["recent_failure_counters"] = {
            "late_correction_failures": 4,
            "rollup_refresh_failures": 3,
            "schema_migration_regressions": 2,
        }
        metadata["trace_dependency_health"] = {
            "schema_alignment": "heldout_alias_family_active",
            "temporal_backfill": "late_batches_pending",
            "rollup_state": "rollup_stale",
            "temporal_closure": "replay_window_open",
        }
        metadata["operational_trace_summary"] = (
            "Bundled DAG-task and warehouse traces capture unseen temporal alias families, "
            "heavier replay pressure, and stale rollups across larger recovery windows."
        )

    return IncidentFixture(
        broken_tables={
            "source_orders": source_broken,
            "catalog": catalog_broken,
            "hourly_rollup": rollup_broken,
        },
        ground_truth_tables={
            "source_orders": source_truth,
            "catalog": catalog_truth,
            "hourly_rollup": rollup_truth,
        },
        metadata=metadata,
    )
