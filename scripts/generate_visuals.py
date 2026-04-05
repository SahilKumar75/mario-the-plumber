#!/usr/bin/env python3
"""Generate benchmark-wide visuals for the README."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mario-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/mario-fontconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"


def _load_json(name: str) -> dict:
    return json.loads((ASSETS / name).read_text(encoding="utf-8"))


def build_benchmark_overview() -> None:
    report = _load_json("benchmark_runs.json")
    rows = report["rows"]
    tasks = [f"task_{index}" for index in range(1, 6)]
    task_labels = ["Task 1", "Task 2", "Task 3", "Task 4", "Task 5"]
    row_order = [
        ("random", "train", "#94a3b8"),
        ("heuristic", "train", "#2563eb"),
        ("trained", "train", "#7c3aed"),
        ("random", "eval", "#cbd5e1"),
        ("heuristic", "eval", "#0f766e"),
        ("trained", "eval", "#ea580c"),
    ]
    # Only plot rows that actually exist in the report
    available = {(item["policy"], item["split"]) for item in rows}
    row_order = [(p, s, c) for p, s, c in row_order if (p, s) in available]

    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(tasks))
    width = 0.13
    center_offset = (len(row_order) - 1) / 2

    for idx, (policy, split, color) in enumerate(row_order):
        row = next(item for item in rows if item["policy"] == policy and item["split"] == split)
        offset = (idx - center_offset) * width
        values = [row[task] for task in tasks]
        ax.bar(x + offset, values, width=width, color=color, label=f"{policy}-{split}")

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.set_ylabel("Average score")
    ax.set_title("ETL Incident Recovery Performance Overview")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(ASSETS / "benchmark_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)



def build_task_gap_chart() -> None:
    report = _load_json("benchmark_runs.json")
    metadata = _load_json("benchmark_metadata.json")
    rows = report["rows"]
    initial_stats = metadata["initial_score_stats"]
    tasks = [1, 2, 3, 4, 5]
    labels = [f"Task {task_id}" for task_id in tasks]
    available_policies = {(item["policy"], item["split"]) for item in rows}

    initial = []
    random_eval = [] if ("random", "eval") in available_policies else None
    heuristic_eval = [] if ("heuristic", "eval") in available_policies else None
    trained_eval = [] if ("trained", "eval") in available_policies else None

    for task_id in tasks:
        train_mean = initial_stats["train"][f"task_{task_id}"]["initial_score_mean"]
        eval_mean = initial_stats["eval"][f"task_{task_id}"]["initial_score_mean"]
        initial.append((train_mean + eval_mean) / 2.0)
        if random_eval is not None:
            random_row = next(item for item in rows if item["policy"] == "random" and item["split"] == "eval")
            random_eval.append(random_row[f"task_{task_id}"])
        if heuristic_eval is not None:
            heuristic_row = next(item for item in rows if item["policy"] == "heuristic" and item["split"] == "eval")
            heuristic_eval.append(heuristic_row[f"task_{task_id}"])
        if trained_eval is not None:
            trained_row = next(item for item in rows if item["policy"] == "trained" and item["split"] == "eval")
            trained_eval.append(trained_row[f"task_{task_id}"])

    # Count bars to compute offsets dynamically
    bar_series = [("initial state", initial, "#cbd5e1")]
    if random_eval is not None:
        bar_series.append(("random (eval)", random_eval, "#94a3b8"))
    if heuristic_eval is not None:
        bar_series.append(("heuristic (eval)", heuristic_eval, "#0f766e"))
    if trained_eval is not None:
        bar_series.append(("trained (eval)", trained_eval, "#ea580c"))

    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(tasks))
    width = 0.18
    center_offset = (len(bar_series) - 1) / 2
    for idx, (label, values, color) in enumerate(bar_series):
        offset = (idx - center_offset) * width
        ax.bar(x + offset, values, width=width, color=color, label=label)

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Average score")
    ax.set_title("Incident Difficulty Gap by Task")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(ASSETS / "difficulty_gap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)



def build_objective_weights_chart() -> None:
    metadata = _load_json("benchmark_metadata.json")
    weights = metadata["benchmark_metadata"]["objective_weights"]
    task_ids = ["1", "2", "3", "4", "5"]
    labels = ["Task 1", "Task 2", "Task 3", "Task 4", "Task 5"]
    keys = sorted({key for task_id in task_ids for key in weights[task_id]})
    colors = {
        "accuracy": "#0f766e",
        "completeness": "#2563eb",
        "consistency": "#1d4ed8",
        "data_quality": "#2563eb",
        "dependency_consistency": "#0f766e",
        "freshness": "#f59e0b",
        "backlog": "#dc2626",
        "resource_efficiency": "#7c3aed",
        "validity": "#14b8a6",
        "summary_consistency": "#14b8a6",
        "schema_alignment": "#1d4ed8",
        "temporal_backfill": "#ea580c",
        "rollup_consistency": "#0f766e",
    }

    fig, ax = plt.subplots(figsize=(12.2, 5.8))
    x = np.arange(len(task_ids))
    bottom = np.zeros(len(task_ids))
    for key in keys:
        values = np.array([weights[task_id].get(key, 0.0) for task_id in task_ids])
        if np.allclose(values, 0.0):
            continue
        ax.bar(x, values, bottom=bottom, color=colors.get(key, "#64748b"), label=key.replace("_", " "))
        bottom += values

    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Weight")
    ax.set_title("Task Scoring Weights Across the ETL Incident Suite")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(ASSETS / "objective_weights.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    build_benchmark_overview()
    build_task_gap_chart()
    build_objective_weights_chart()


if __name__ == "__main__":
    main()
