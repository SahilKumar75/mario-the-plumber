from __future__ import annotations

import pandas as pd

from .shared import IncidentFixture


def load_task5_fixture(profile: str, split: str) -> IncidentFixture:
    catalog_truth = pd.DataFrame(
        {
            "product_id": [601, 602, 603, 604],
            "product_name": ["Valve", "Sensor", "Pump", "Controller"],
            "unit_price": [18.0, 52.0, 77.0, 112.0],
            "category": ["hardware", "iot", "hardware", "iot"],
        }
    )
    source_truth = pd.DataFrame(
        {
            "order_id": [11001, 11002, 11003, 11004, 11005, 11006, 11007, 11008, 11009, 11010, 11011, 11012, 11013, 11014],
            "batch_id": ["t1", "t1", "t1", "t2", "t2", "t2", "t3", "t3", "t4", "t4", "t5", "t5", "t5", "t5"],
            "product_id": [601, 602, 604, 603, 601, 602, 604, 603, 601, 604, 602, 603, 601, 604],
            "quantity": [2, 1, 3, 2, 4, 1, 2, 3, 1, 5, 2, 4, 3, 2],
            "event_ts": [
                "2026-03-29T10:00:00Z",
                "2026-03-29T10:30:00Z",
                "2026-03-29T11:00:00Z",
                "2026-03-29T12:00:00Z",
                "2026-03-29T12:20:00Z",
                "2026-03-29T13:00:00Z",
                "2026-03-29T14:00:00Z",
                "2026-03-29T14:20:00Z",
                "2026-03-29T15:00:00Z",
                "2026-03-29T15:30:00Z",
                "2026-03-29T14:40:00Z",
                "2026-03-29T14:50:00Z",
                "2026-03-29T15:10:00Z",
                "2026-03-29T15:45:00Z",
            ],
        }
    )
    source_truth = source_truth.merge(catalog_truth[["product_id", "unit_price"]], on="product_id", how="left")
    source_truth["gross_revenue"] = source_truth["quantity"] * source_truth["unit_price"]
    source_truth = source_truth.drop(columns=["unit_price"])
    rollup_truth = (
        source_truth.assign(hour_bucket=pd.to_datetime(source_truth["event_ts"]).dt.strftime("%Y-%m-%dT%H:00:00Z"))
        .groupby("hour_bucket", as_index=False)
        .agg(order_count=("order_id", "count"), gross_revenue=("gross_revenue", "sum"))
    )

    pending_orders = source_truth[source_truth["batch_id"].isin(["t3", "t4", "t5"])].copy()
    visible_orders = source_truth[source_truth["batch_id"].isin(["t1", "t2"])].copy()
    source_broken = visible_orders.copy()
    catalog_broken = catalog_truth.copy()
    rollup_broken = rollup_truth[rollup_truth["hour_bucket"] < "2026-03-29T14:00:00Z"].copy()

    if profile == "temporal_rollup_backfill_incident":
        source_broken["product_id"] = source_broken["product_id"].astype(str)
        source_broken["quantity"] = ["2 units", "1", "3 units", "2", "4", "1 units"]
        source_broken["event_ts"] = [
            "29/03/2026 10:00",
            "2026-03-29T10:30:00Z",
            "2026-03-29 11:00:00+05:30",
            "29/03/2026 12:00",
            "2026-03-29T12:20:00Z",
            "2026-03-29 13:00:00+05:30",
        ]
        source_broken["gross_revenue"] = (visible_orders["gross_revenue"] * 0.83).round(2).astype(object)
        catalog_broken["unit_price"] = ["$18.00", "$52.00", "7700 cents", "$112.00"]
        rollup_broken.loc[:, "gross_revenue"] = (rollup_broken["gross_revenue"] * 0.88).round(2)
        rollup_broken.loc[rollup_broken.index[0], "hour_bucket"] = "29/03/2026 10:00"
    elif profile == "schema_evolution_backfill_recovery":
        source_broken["product_id"] = source_broken["product_id"].astype(str)
        source_broken["quantity"] = ["2 units", "1", "3 units", "2", "missing", "1 units"]
        source_broken = source_broken.rename(columns={"event_ts": "observed_at"})
        source_broken["observed_at"] = [
            "29/03/2026 10:00",
            "2026-03-29T10:30:00Z",
            "2026-03-29 11:00:00+05:30",
            "29/03/2026 12:00",
            "2026-03-29T12:20:00Z",
            "2026-03-29 13:00:00+05:30",
        ]
        source_broken["gross_revenue"] = (visible_orders["gross_revenue"] * 0.83).round(2).astype(object)
        catalog_broken["unit_price"] = ["$18.00", "5200 cents", "7700 cents", "$112.00"]
        catalog_broken = catalog_broken.rename(columns={"category": "product_segment"})
        catalog_broken.loc[1, "product_segment"] = " IoT "
        catalog_broken = pd.concat([catalog_broken, catalog_broken.iloc[[2]]], ignore_index=True)
        rollup_broken.loc[:, "gross_revenue"] = (rollup_broken["gross_revenue"] * 0.88).round(2)
        rollup_broken.loc[rollup_broken.index[0], "hour_bucket"] = "29/03/2026 10:00"
    elif profile == "late_correction_backpressure_incident":
        source_broken["product_id"] = source_broken["product_id"].astype(str)
        source_broken["quantity"] = ["2", "1 units", "3", "2 units", "4", "missing"]
        source_broken["event_ts"] = [
            "29/03/2026 10:00",
            "2026-03-29T10:30:00Z",
            "2026-03-29 11:00:00+05:30",
            "29/03/2026 12:00",
            "2026-03-29T12:20:00Z",
            "2026-03-29 13:00:00+05:30",
        ]
        source_broken["gross_revenue"] = (visible_orders["gross_revenue"] * 0.83).round(2).astype(object)
        catalog_broken["unit_price"] = ["$18.00", "$52.00", "7700 cents", "$112.00"]
        catalog_broken.loc[1, "category"] = " IoT "
        catalog_broken = pd.concat([catalog_broken, catalog_broken.iloc[[2]]], ignore_index=True)
        rollup_broken.loc[:, "gross_revenue"] = (rollup_broken["gross_revenue"] * 0.88).round(2)
        rollup_broken = rollup_broken.rename(columns={"hour_bucket": "window_start"})
        rollup_broken.loc[rollup_broken.index[0], "window_start"] = "29/03/2026 10:00"
    elif profile == "heldout_temporal_schema_extension_family":
        source_broken["product_id"] = source_broken["product_id"].astype(str)
        source_broken["quantity"] = ["2 units", "missing", "3 units", "2", "4", "1 units"]
        pending_orders = pending_orders.rename(
            columns={"event_ts": "source_snapshot_utc", "product_id": "sku_key", "quantity": "ordered_units"}
        )
        pending_orders["sku_key"] = pending_orders["sku_key"].astype(str)
        source_broken = source_broken.rename(
            columns={"event_ts": "source_snapshot_utc", "product_id": "sku_key", "quantity": "ordered_units"}
        )
        source_broken["source_snapshot_utc"] = [
            "29/03/2026 10:00",
            "2026-03-29T10:30:00Z",
            "2026-03-29 11:00:00+05:30",
            "29/03/2026 12:00",
            "2026-03-29T12:20:00Z",
            "2026-03-29 13:00:00+05:30",
        ]
        source_broken["recognized_revenue_value"] = (visible_orders["gross_revenue"] * 0.77).round(2).astype(object)
        source_broken = source_broken.drop(columns=["gross_revenue"])
        catalog_broken["unit_price"] = ["$18.00", "5200 cents", "7700 cents", "$112.00"]
        catalog_broken = catalog_broken.rename(columns={"category": "catalog_family_group"})
        catalog_broken.loc[1, "catalog_family_group"] = " IoT "
        catalog_broken = pd.concat([catalog_broken, catalog_broken.iloc[[2]]], ignore_index=True)
        rollup_broken = rollup_broken.rename(
            columns={"hour_bucket": "rollup_bucket_start_utc", "gross_revenue": "recognized_revenue_value"}
        )
        rollup_broken.loc[:, "recognized_revenue_value"] = (rollup_broken["recognized_revenue_value"] * 0.83).round(2)
        rollup_broken.loc[rollup_broken.index[0], "rollup_bucket_start_utc"] = "29/03/2026 10:00"
    elif profile == "heldout_temporal_rollup_contract_family":
        source_broken["product_id"] = source_broken["product_id"].astype(str)
        source_broken["quantity"] = ["2 units", "1", "3 units", "2", "missing", "1 units"]
        pending_orders = pending_orders.rename(columns={"event_ts": "observed_watermark_at", "product_id": "inventory_sku_key"})
        pending_orders["inventory_sku_key"] = pending_orders["inventory_sku_key"].astype(str)
        source_broken = source_broken.rename(columns={"event_ts": "observed_watermark_at", "product_id": "inventory_sku_key"})
        source_broken["observed_watermark_at"] = [
            "29/03/2026 10:00",
            "2026-03-29T10:30:00Z",
            "2026-03-29 11:00:00+05:30",
            "29/03/2026 12:00",
            "2026-03-29T12:20:00Z",
            "2026-03-29 13:00:00+05:30",
        ]
        source_broken["booked_revenue_usd"] = (visible_orders["gross_revenue"] * 0.79).round(2).astype(object)
        source_broken = source_broken.drop(columns=["gross_revenue"])
        catalog_broken["unit_price"] = ["$18.00", "$52.00", "7700 cents", "$112.00"]
        catalog_broken = catalog_broken.rename(columns={"unit_price": "unit_cost_cents"})
        rollup_broken = rollup_broken.rename(
            columns={"hour_bucket": "rollup_window_start_utc", "gross_revenue": "net_sales_value"}
        )
        rollup_broken.loc[:, "net_sales_value"] = (rollup_broken["net_sales_value"] * 0.84).round(2)
        rollup_broken.loc[rollup_broken.index[0], "rollup_window_start_utc"] = "29/03/2026 10:00"
    elif profile == "heldout_temporal_correction_replay_family":
        source_broken["product_id"] = source_broken["product_id"].astype(str)
        source_broken["quantity"] = ["2 units", "1", "3 units", "2", "missing", "1 units"]
        pending_orders = pending_orders.rename(columns={"event_ts": "replay_observed_at", "product_id": "item_sku"})
        pending_orders["item_sku"] = pending_orders["item_sku"].astype(str)
        source_broken = source_broken.rename(columns={"event_ts": "replay_observed_at", "product_id": "item_sku"})
        source_broken["replay_observed_at"] = [
            "29/03/2026 10:00",
            "2026-03-29T10:30:00Z",
            "2026-03-29 11:00:00+05:30",
            "29/03/2026 12:00",
            "2026-03-29T12:20:00Z",
            "2026-03-29 13:00:00+05:30",
        ]
        source_broken["revenue_usd_value"] = (visible_orders["gross_revenue"] * 0.76).round(2).astype(object)
        source_broken = source_broken.drop(columns=["gross_revenue"])
        catalog_broken["unit_price"] = ["$18.00", "5200 cents", "7700 cents", "$112.00"]
        catalog_broken = catalog_broken.rename(columns={"category": "catalog_segment_name"})
        catalog_broken.loc[1, "catalog_segment_name"] = " IoT "
        catalog_broken = pd.concat([catalog_broken, catalog_broken.iloc[[2]]], ignore_index=True)
        rollup_broken = rollup_broken.rename(
            columns={"hour_bucket": "bucket_window_utc_start", "gross_revenue": "replayed_revenue_value"}
        )
        rollup_broken.loc[:, "replayed_revenue_value"] = (rollup_broken["replayed_revenue_value"] * 0.82).round(2)
        rollup_broken.loc[rollup_broken.index[0], "bucket_window_utc_start"] = "29/03/2026 10:00"

    profile_manifest = {
        "schema_evolution_backfill_recovery": {
            "profile_family": "familiar_temporal",
            "novelty_axes": ["schema_alias_shift", "catalog_contract_drift"],
            "failed_tasks": ["sync_catalog", "replay_late_batches", "refresh_hourly_rollup"],
            "affected_hour_buckets": ["2026-03-29T14:00:00Z", "2026-03-29T15:00:00Z"],
            "late_correction_order_ids": [11007, 11008, 11009, 11010, 11011, 11012, 11013, 11014],
            "watermark_before": "2026-03-29T13:00:00Z",
            "expected_watermark_after_replay": "2026-03-29T15:45:00Z",
            "source_contract_version": "v3-source-alias",
            "rollup_contract_version": "v2-hourly-rollup",
            "downstream_assets": ["hourly_rollup", "revenue_monitor"],
        },
        "late_correction_backpressure_incident": {
            "profile_family": "familiar_temporal",
            "novelty_axes": ["late_correction_replay", "rollup_window_alias"],
            "failed_tasks": ["replay_late_batches", "refresh_hourly_rollup"],
            "affected_hour_buckets": ["2026-03-29T14:00:00Z", "2026-03-29T15:00:00Z"],
            "late_correction_order_ids": [11007, 11008, 11009, 11010, 11011, 11012, 11013, 11014],
            "watermark_before": "2026-03-29T14:20:00Z",
            "expected_watermark_after_replay": "2026-03-29T15:45:00Z",
            "source_contract_version": "v2-source-orders",
            "rollup_contract_version": "v2-window-start-alias",
            "downstream_assets": ["hourly_rollup", "revenue_monitor"],
        },
        "heldout_temporal_schema_extension_family": {
            "profile_family": "heldout_temporal",
            "novelty_axes": ["unknown_source_aliases", "schema_extension", "catalog_contract_drift"],
            "failed_tasks": ["load_source_orders", "sync_catalog", "replay_late_batches", "refresh_hourly_rollup"],
            "affected_hour_buckets": ["2026-03-29T14:00:00Z", "2026-03-29T15:00:00Z"],
            "late_correction_order_ids": [11007, 11008, 11009, 11010, 11011, 11012, 11013, 11014],
            "watermark_before": "2026-03-29T13:00:00Z",
            "expected_watermark_after_replay": "2026-03-29T15:45:00Z",
            "source_contract_version": "v4-source-schema-extension",
            "rollup_contract_version": "v3-window-start-gross-sales",
            "downstream_assets": ["hourly_rollup", "revenue_monitor", "ops_watermark_audit"],
        },
        "heldout_temporal_rollup_contract_family": {
            "profile_family": "heldout_temporal",
            "novelty_axes": ["unknown_catalog_aliases", "rollup_contract_drift", "backfill_pressure"],
            "failed_tasks": ["sync_catalog", "replay_late_batches", "refresh_hourly_rollup"],
            "affected_hour_buckets": ["2026-03-29T13:00:00Z", "2026-03-29T14:00:00Z", "2026-03-29T15:00:00Z"],
            "late_correction_order_ids": [11007, 11008, 11009, 11010, 11011, 11012, 11013, 11014],
            "watermark_before": "2026-03-29T12:20:00Z",
            "expected_watermark_after_replay": "2026-03-29T15:45:00Z",
            "source_contract_version": "v3-observed-at",
            "rollup_contract_version": "v4-rollup-revenue-total",
            "downstream_assets": ["hourly_rollup", "revenue_monitor", "margin_dashboard"],
        },
        "heldout_temporal_correction_replay_family": {
            "profile_family": "heldout_temporal",
            "novelty_axes": ["unknown_source_aliases", "late_correction_replay", "catalog_contract_drift"],
            "failed_tasks": ["load_source_orders", "replay_late_batches", "refresh_hourly_rollup"],
            "affected_hour_buckets": ["2026-03-29T14:00:00Z", "2026-03-29T15:00:00Z"],
            "late_correction_order_ids": [11007, 11008, 11009, 11010, 11011, 11012, 11013, 11014],
            "watermark_before": "2026-03-29T13:00:00Z",
            "expected_watermark_after_replay": "2026-03-29T15:45:00Z",
            "source_contract_version": "v4-source-sku-time",
            "rollup_contract_version": "v3-window-start-gross-sales",
            "downstream_assets": ["hourly_rollup", "revenue_monitor", "correction_audit_feed"],
        },
    }[profile]

    backlog_rows = len(pending_orders)
    required_resource_level = 3 if backlog_rows >= 4 else 2
    freshness_lag_minutes = 180 if split == "eval" else 120
    if profile_manifest["profile_family"] == "heldout_temporal":
        freshness_lag_minutes += 30
    backlog_age_minutes = 240 if split == "train" else 360
    if profile_manifest["profile_family"] == "heldout_temporal":
        backlog_age_minutes += 60
    incident_id = f"t5-{profile}-{split}"
    metadata = {
        "pending_orders": pending_orders,
        "backlog_rows": backlog_rows,
        "queue_backlog_age_minutes": backlog_age_minutes,
        "freshness_lag_minutes": freshness_lag_minutes,
        "resource_level": 1,
        "required_resource_level": required_resource_level,
        "pending_batches": 3 if backlog_rows > 0 else 0,
        "downstream_stale": True,
        "workload_pressure": 0.95 if split == "eval" else 0.8,
        "incident_manifest": {
            "incident_id": incident_id,
            "dag_id": "temporal_orders_rollup",
            "warehouse": "finance_wh",
            "severity": "critical" if split == "eval" else "high",
            "failed_tasks": profile_manifest["failed_tasks"],
            "downstream_assets": profile_manifest["downstream_assets"],
            "profile_family": profile_manifest["profile_family"],
            "novelty_axes": profile_manifest["novelty_axes"],
            "affected_hour_buckets": profile_manifest["affected_hour_buckets"],
            "late_correction_order_ids": profile_manifest["late_correction_order_ids"],
            "watermark_before": profile_manifest["watermark_before"],
            "expected_watermark_after_replay": profile_manifest["expected_watermark_after_replay"],
            "source_contract_version": profile_manifest["source_contract_version"],
            "rollup_contract_version": profile_manifest["rollup_contract_version"],
        },
        "dag_runs": pd.DataFrame(
            [
                {"run_id": f"{incident_id}-1", "task_id": "load_source_orders", "state": "failed" if "load_source_orders" in profile_manifest["failed_tasks"] else "success", "attempt": 2},
                {"run_id": f"{incident_id}-1", "task_id": "sync_catalog", "state": "failed" if "sync_catalog" in profile_manifest["failed_tasks"] else "success", "attempt": 1},
                {"run_id": f"{incident_id}-1", "task_id": "replay_late_batches", "state": "failed", "attempt": 1},
                {"run_id": f"{incident_id}-1", "task_id": "refresh_hourly_rollup", "state": "failed", "attempt": 1},
            ]
        ),
        "warehouse_events": pd.DataFrame(
            [
                {"ts": "2026-03-29T15:40:00Z", "system": "airflow", "event": "late_batch_replay_blocked"},
                {"ts": "2026-03-29T15:42:00Z", "system": "warehouse", "event": "hourly_rollup_stale"},
                {"ts": "2026-03-29T15:45:00Z", "system": "schema_registry", "event": "temporal_contract_shift_detected"},
                {"ts": "2026-03-29T15:47:00Z", "system": "watermark_monitor", "event": f"watermark_stalled_at:{profile_manifest['watermark_before']}"},
            ]
        ),
        "trace_drift_markers": [
            "temporal_contract_shift",
            "late_batch_replay_backpressure",
            "hourly_rollup_consistency_alarm",
        ],
        "trace_dependency_health": {
            "schema_alignment": "schema_drift_active",
            "temporal_backfill": "late_batches_pending",
            "rollup_state": "rollup_stale",
            "temporal_closure": "replay_window_open",
        },
        "recent_failure_counters": {
            "late_correction_failures": 2,
            "rollup_refresh_failures": 1,
            "schema_migration_regressions": 1,
        },
        "operational_trace_summary": (
            "Bundled DAG-task and warehouse traces capture replay windows, stalled watermarks, "
            "contract-version drift, and stale temporal rollups."
        ),
    }
    if profile_manifest["profile_family"] == "heldout_temporal":
        metadata["recent_failure_counters"] = {
            "late_correction_failures": 3,
            "rollup_refresh_failures": 2,
            "schema_migration_regressions": 2,
        }
        metadata["trace_dependency_health"] = {
            "schema_alignment": "heldout_alias_family_active",
            "temporal_backfill": "late_batches_pending",
            "rollup_state": "rollup_stale",
        }
        metadata["operational_trace_summary"] = (
            "Bundled DAG-task and warehouse traces capture unseen temporal alias families, replay pressure, and stale rollups."
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
