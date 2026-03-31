#!/usr/bin/env python3
"""Benchmark Mario the Plumber across splits and baseline policies."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
from statistics import mean, pstdev
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import run_baseline
from models import PipelineDoctorAction
from server.data_generator import MAX_STEPS, TASK_THRESHOLDS
from server.pipeline_doctor_environment import PipelineDoctorEnvironment


def run_random_baseline(seed: int, *, split: str = "train") -> dict[str, Any]:
    rng = random.Random(seed)
    results: list[dict[str, Any]] = []

    for task_id in sorted(TASK_THRESHOLDS):
        env = PipelineDoctorEnvironment()
        observation = env.reset(seed=seed, task_id=task_id, split=split)

        for _ in range(MAX_STEPS[task_id]):
            if env.state.done:
                break
            action = _random_action(env, observation, rng)
            observation = env.step(action)
            if env.state.done:
                break

        if not env.state.done:
            observation = env.step(PipelineDoctorAction(action_id=15))

        results.append(
            {
                "task_id": task_id,
                "score": round(observation.current_score, 4),
                "steps": env.state.step_count,
                "success": bool(env.state.success),
            }
        )

    return {
        "status": "complete",
        "policy_mode": "random",
        "scenario_split": split,
        "results": results,
        "average_score": round(mean(float(item["score"]) for item in results), 4),
    }


def _random_action(
    env: PipelineDoctorEnvironment,
    observation: Any,
    rng: random.Random,
) -> PipelineDoctorAction:
    action_id = rng.randint(0, 19)
    current_columns = list(env._current_frame().columns)  # noqa: SLF001

    if action_id in {3, 4, 5, 6, 7, 8, 9, 11, 12} and current_columns:
        target_column = rng.choice(current_columns)
    elif action_id == 0:
        if env.state.task_id == 3 and env.state.active_table in {"orders", "customers"}:
            target_column = "customers" if env.state.active_table == "orders" else "products"
        elif env.state.task_id == 4 and env.state.active_table in {"orders", "products"}:
            target_column = "products" if env.state.active_table == "orders" else "daily_summary"
        else:
            target_column = None
    else:
        target_column = None

    new_name = None
    if action_id == 12 and target_column:
        new_name = f"{target_column}_v{rng.randint(1, 3)}"

    column_order = None
    if action_id == 13:
        column_order = current_columns[:]
        rng.shuffle(column_order)

    return PipelineDoctorAction(
        action_id=action_id,
        target_column=target_column,
        new_name=new_name,
        column_order=column_order,
    )


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "average_score_mean": round(mean(run["average_score"] for run in runs), 4),
        "average_score_std": round(pstdev(run["average_score"] for run in runs), 4)
        if len(runs) > 1
        else 0.0,
        "task_means": {},
        "task_stds": {},
    }
    for task_id in sorted(TASK_THRESHOLDS):
        scores = [
            next(item["score"] for item in run["results"] if item["task_id"] == task_id)
            for run in runs
        ]
        summary["task_means"][f"task_{task_id}"] = round(mean(scores), 4)
        summary["task_stds"][f"task_{task_id}"] = round(pstdev(scores), 4) if len(scores) > 1 else 0.0
    return summary


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "| Policy | Split | Avg Score | Task 1 | Task 2 | Task 3 | Task 4 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['policy']} | {row['split']} | {row['average_score_mean']:.4f} | "
            f"{row['task_1']:.4f} | {row['task_2']:.4f} | {row['task_3']:.4f} | {row['task_4']:.4f} |"
        )
    return "\n".join(lines)


def write_csv(report: dict[str, Any], path: str) -> None:
    fieldnames = [
        "policy",
        "split",
        "average_score_mean",
        "average_score_std",
        "task_1",
        "task_1_std",
        "task_2",
        "task_2_std",
        "task_3",
        "task_3_std",
        "task_4",
        "task_4_std",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["rows"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Mario the Plumber baselines.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--splits", nargs="+", default=["train", "eval"])
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["random", "heuristic", "hybrid"],
        choices=["random", "heuristic", "hybrid", "pure-llm"],
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--csv-out", default=None)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    raw_runs: dict[str, Any] = {}

    for split in args.splits:
        for policy in args.policies:
            runs = []
            for seed in args.seeds:
                if policy == "random":
                    run = run_random_baseline(seed, split=split)
                else:
                    run = run_baseline(
                        seed=seed,
                        split=split,
                        policy_mode=policy,
                        model_name=args.model_name,
                    )
                runs.append(run)
            summary = summarize_runs(runs)
            rows.append(
                {
                    "policy": policy,
                    "split": split,
                    "average_score_mean": summary["average_score_mean"],
                    "average_score_std": summary["average_score_std"],
                    "task_1": summary["task_means"]["task_1"],
                    "task_1_std": summary["task_stds"]["task_1"],
                    "task_2": summary["task_means"]["task_2"],
                    "task_2_std": summary["task_stds"]["task_2"],
                    "task_3": summary["task_means"]["task_3"],
                    "task_3_std": summary["task_stds"]["task_3"],
                    "task_4": summary["task_means"]["task_4"],
                    "task_4_std": summary["task_stds"]["task_4"],
                }
            )
            raw_runs[f"{policy}:{split}"] = runs

    report = {"rows": rows, "runs": raw_runs}
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.csv_out:
        write_csv(report, args.csv_out)
    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(to_markdown(report))


if __name__ == "__main__":
    main()
