from __future__ import annotations

from datetime import datetime

import pandas as pd


def duplicate_row_count(frame: pd.DataFrame) -> int:
    key_column = _primary_key_column(frame)
    if key_column and key_column in frame.columns:
        return int(frame.duplicated(subset=[key_column]).sum())
    return int(frame.duplicated().sum())


def score_single_table(
    fixed_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    expected_types: dict[str, str],
) -> tuple[float, dict[str, float]]:
    total_cells = max(len(fixed_df) * max(len(fixed_df.columns), 1), 1)
    completeness = 1.0 - (fixed_df.isnull().sum().sum() / total_cells)

    validity_matches = sum(
        1
        for column in fixed_df.columns
        if str(fixed_df[column].dtype) == expected_types.get(column, "")
    )
    validity = validity_matches / max(len(fixed_df.columns), 1)

    consistency = 1.0
    if len(fixed_df) > 0:
        consistency = 1.0 - (duplicate_row_count(fixed_df) / len(fixed_df))

    accuracy = _accuracy(fixed_df, ground_truth_df)
    score = round(
        (0.20 * completeness)
        + (0.20 * validity)
        + (0.30 * consistency)
        + (0.30 * accuracy),
        4,
    )

    return score, {
        "completeness": round(completeness, 4),
        "validity": round(validity, 4),
        "consistency": round(consistency, 4),
        "accuracy": round(accuracy, 4),
    }


def _canonical_event_date(value: object) -> str | None:
    if pd.isna(value):
        return None
    parsed = _coerce_utc_timestamp(value)
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d")


def _canonical_hour_bucket(value: object) -> str | None:
    if pd.isna(value):
        return None
    parsed = _coerce_utc_timestamp(value)
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%dT%H:00:00Z")


def _accuracy(fixed_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> float:
    if list(fixed_df.columns) != list(ground_truth_df.columns):
        return 0.0
    if len(fixed_df) != len(ground_truth_df):
        return 0.0
    if len(fixed_df) == 0:
        return 1.0
    fixed = fixed_df.astype(object).where(fixed_df.notna(), "__nan__")
    ground_truth = ground_truth_df.astype(object).where(ground_truth_df.notna(), "__nan__")
    matches = (fixed == ground_truth).all(axis=1)
    return float(matches.mean())


def _coerce_utc_timestamp(value: object) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT

    text = str(value).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%d/%m/%Y %H:%M", "%m-%d-%Y %H:%M", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(text, fmt)
            return pd.Timestamp(parsed, tz="UTC")
        except (TypeError, ValueError):
            continue

    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.isna(parsed):
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.isna(parsed):
            return pd.NaT
        if parsed.tzinfo is None:
            return parsed.tz_localize("UTC")
        return parsed.tz_convert("UTC")
    return parsed


def _primary_key_column(frame: pd.DataFrame) -> str | None:
    for candidate in ("transaction_id", "order_id", "customer_id", "product_id"):
        if candidate in frame.columns:
            return candidate
    return None
