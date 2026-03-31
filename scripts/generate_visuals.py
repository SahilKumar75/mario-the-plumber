#!/usr/bin/env python3
"""Generate README visuals from the latest benchmark artifacts."""

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
        ("random", "eval", "#cbd5e1"),
        ("heuristic", "eval", "#0f766e"),
    ]

    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(tasks))
    width = 0.18

    for idx, (policy, split, color) in enumerate(row_order):
        row = next(item for item in rows if item["policy"] == policy and item["split"] == split)
        offset = (idx - 1.5) * width
        values = [row[task] for task in tasks]
        ax.bar(x + offset, values, width=width, color=color, label=f"{policy}-{split}")

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.set_ylabel("Average score")
    ax.set_title("Mario Benchmark Overview")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(ASSETS / "benchmark_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_task_difficulty_profile() -> None:
    metadata = _load_json("benchmark_metadata.json")
    train = metadata["initial_score_stats"]["train"]
    eval_ = metadata["initial_score_stats"]["eval"]
    labels = [train[f"task_{index}"]["name"].replace("Temporal ETL Recovery with Reward Machine Structure", "Task 5") for index in range(1, 6)]
    train_scores = [train[f"task_{index}"]["initial_score_mean"] for index in range(1, 6)]
    eval_scores = [eval_[f"task_{index}"]["initial_score_mean"] for index in range(1, 6)]

    fig, ax = plt.subplots(figsize=(11, 5.8))
    y = np.arange(len(labels))
    ax.barh(y + 0.18, train_scores, height=0.34, color="#2563eb", label="train")
    ax.barh(y - 0.18, eval_scores, height=0.34, color="#0f766e", label="eval")
    ax.set_xlim(0, 1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Initial score before any repair")
    ax.set_title("Starting Difficulty by Task")
    ax.grid(axis="x", alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "task_difficulty_profile.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_task5_adaptation() -> None:
    report = _load_json("adaptation_report.json")
    labels = ["Task 5 train", "Task 5 eval", "Held-out eval family"]
    means = [
        report["train_task5"]["mean"],
        report["eval_task5"]["mean"],
        report["heldout_profile_family_task5"]["mean"],
    ]
    mins = [
        report["train_task5"]["min"],
        report["eval_task5"]["min"],
        report["heldout_profile_family_task5"]["min"],
    ]
    maxes = [
        report["train_task5"]["max"],
        report["eval_task5"]["max"],
        report["heldout_profile_family_task5"]["max"],
    ]
    lowers = [mean - min_ for mean, min_ in zip(means, mins, strict=True)]
    uppers = [max_ - mean for mean, max_ in zip(means, maxes, strict=True)]

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, color=["#2563eb", "#0f766e", "#7c3aed"], width=0.56)
    ax.errorbar(
        x,
        means,
        yerr=[lowers, uppers],
        fmt="none",
        ecolor="#111827",
        elinewidth=1.5,
        capsize=5,
    )
    for bar, value in zip(bars, means, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Average score")
    ax.set_title("Task 5 Held-out Adaptation Snapshot")
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(ASSETS / "task5_adaptation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    build_benchmark_overview()
    build_task_difficulty_profile()
    build_task5_adaptation()


if __name__ == "__main__":
    main()
