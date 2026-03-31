#!/usr/bin/env python3
"""Generate static benchmark visuals for the Mario the Plumber README."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mario-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/mario-fontconfig")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from inference import _next_table, _table_should_advance, _task4_heuristic_action
from models import PipelineDoctorAction
from server.pipeline_doctor_environment import PipelineDoctorEnvironment
ASSETS = ROOT / "docs" / "assets"


def build_benchmark_landscape() -> None:
    data = pd.DataFrame(
        [
            ("random", "train", 0.6883, 0.5433, 0.1952, 0.2988),
            ("heuristic", "train", 0.9125, 0.9667, 0.9320, 0.8000),
            ("random", "eval", 0.6883, 0.5433, 0.1952, 0.2844),
            ("heuristic", "eval", 0.9125, 0.9667, 0.9570, 0.8000),
        ],
        columns=["policy", "split", "task_1", "task_2", "task_3", "task_4"],
    )
    task_columns = ["task_1", "task_2", "task_3", "task_4"]
    labels = ["Task 1", "Task 2", "Task 3", "Task 4"]
    colors = {
        ("random", "train"): "#94a3b8",
        ("heuristic", "train"): "#2563eb",
        ("random", "eval"): "#cbd5e1",
        ("heuristic", "eval"): "#0f766e",
    }

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    x = range(len(task_columns))
    width = 0.18

    for index, row in enumerate(data.itertuples(index=False)):
        offset = (index - 1.5) * width
        values = [getattr(row, column) for column in task_columns]
        ax.bar(
            [position + offset for position in x],
            values,
            width=width,
            label=f"{row.policy}-{row.split}",
            color=colors[(row.policy, row.split)],
        )

    ax.set_ylim(0, 1.05)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Average Score")
    ax.set_title("Mario Benchmark Landscape")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(ASSETS / "benchmark_landscape.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_task4_recovery_curve() -> None:
    env = PipelineDoctorEnvironment()
    observation = env.reset(seed=42, task_id=4, split="train")
    history = [
        {
            "step": 0,
            "score": observation.current_score,
            "backlog_rows": observation.backlog_rows,
            "resource_level": observation.resource_level,
            "freshness_lag": observation.freshness_lag_minutes,
        }
    ]

    for _ in range(20):
        action = _task4_heuristic_action(observation)
        observation = env.step(action)
        history.append(
            {
                "step": env.state.step_count,
                "score": observation.current_score,
                "backlog_rows": observation.backlog_rows,
                "resource_level": observation.resource_level,
                "freshness_lag": observation.freshness_lag_minutes,
            }
        )
        if env.state.done:
            break
        if _table_should_advance(4, env, observation):
            next_table = _next_table(env.state.active_table, task_id=4)
            if next_table:
                observation = env.step(PipelineDoctorAction(action_id=0, target_column=next_table))
                history.append(
                    {
                        "step": env.state.step_count,
                        "score": observation.current_score,
                        "backlog_rows": observation.backlog_rows,
                        "resource_level": observation.resource_level,
                        "freshness_lag": observation.freshness_lag_minutes,
                    }
                )
                if env.state.done:
                    break

    frame = pd.DataFrame(history)
    fig, ax1 = plt.subplots(figsize=(10.5, 5.6))
    ax1.plot(frame["step"], frame["score"], color="#2563eb", linewidth=2.5, label="score")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Score", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(frame["step"], frame["backlog_rows"], color="#dc2626", linewidth=2, label="backlog")
    ax2.plot(frame["step"], frame["resource_level"], color="#0f766e", linewidth=2, label="resource level")
    ax2.plot(frame["step"], frame["freshness_lag"] / 30.0, color="#f59e0b", linewidth=2, linestyle="--", label="freshness lag / 30")
    ax2.set_ylabel("Operational Signals", color="#374151")
    ax2.tick_params(axis="y", labelcolor="#374151")

    handles = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in handles]
    ax1.legend(handles, labels, loc="upper left", frameon=False)
    ax1.set_title("Task 4 Incremental Recovery Trajectory")
    fig.tight_layout()
    fig.savefig(ASSETS / "task4_recovery_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    build_benchmark_landscape()
    build_task4_recovery_curve()


if __name__ == "__main__":
    main()
