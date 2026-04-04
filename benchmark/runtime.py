"""Runtime metadata helpers for Mario benchmark modes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    from .catalog import BENCHMARK_VERSION, PROFILE_DESCRIPTIONS, RUNTIME_MODES, TASK_CARDS, benchmark_metadata
except ImportError:
    from benchmark.catalog import BENCHMARK_VERSION, PROFILE_DESCRIPTIONS, RUNTIME_MODES, TASK_CARDS, benchmark_metadata

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"


def _stringify_keys(value):
    if isinstance(value, dict):
        return {str(key): _stringify_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_keys(item) for item in value]
    return value


def runtime_mode() -> str:
    """Return the active benchmark runtime mode."""

    value = os.getenv("MARIO_RUNTIME_MODE", "benchmark").strip().lower()
    if value not in RUNTIME_MODES:
        return "benchmark"
    return value


def runtime_summary() -> dict[str, Any]:
    """Top-level runtime metadata for benchmark-oriented routes and docs."""

    mode = runtime_mode()
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "runtime_mode": mode,
        "runtime_mode_card": RUNTIME_MODES[mode],
        "available_runtime_modes": RUNTIME_MODES,
    }


def benchmark_metadata_payload() -> dict[str, Any]:
    metadata = benchmark_metadata()
    summary = runtime_summary()
    return {
        **metadata,
        "runtime_mode": summary["runtime_mode_card"],
        "runtime_mode_name": summary["runtime_mode"],
        "available_runtime_modes": summary["available_runtime_modes"],
    }


def benchmark_runs_payload() -> dict[str, Any]:
    """Return the latest benchmark artifact payload when available."""

    json_path = ASSETS / "benchmark_runs.json"
    if not json_path.exists():
        return {"available": False, "path": str(json_path)}
    return {"available": True, "path": str(json_path), "report": json.loads(json_path.read_text(encoding="utf-8"))}


def adaptation_payload() -> dict[str, Any]:
    """Return the latest adaptation artifact payload when available."""

    json_path = ASSETS / "adaptation_report.json"
    if not json_path.exists():
        return {"available": False, "path": str(json_path)}
    return {"available": True, "path": str(json_path), "report": json.loads(json_path.read_text(encoding="utf-8"))}


def benchmark_profiles_payload() -> dict[str, Any]:
    """Expose profile families grouped by task and split."""

    metadata = benchmark_metadata()
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "runtime_mode": runtime_mode(),
        "scenario_profiles": _stringify_keys(metadata["scenario_profiles"]),
        "profile_descriptions": PROFILE_DESCRIPTIONS,
        "synthetic_data_notes": metadata["synthetic_data_notes"],
    }


def benchmark_tasks_payload() -> dict[str, Any]:
    """Expose benchmark task cards and formal task structure."""

    metadata = benchmark_metadata()
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "runtime_mode": runtime_mode(),
        "task_cards": _stringify_keys(TASK_CARDS),
        "formal_task_specs": _stringify_keys(metadata["formal_task_specs"]),
        "objective_weights": _stringify_keys(metadata["objective_weights"]),
    }
