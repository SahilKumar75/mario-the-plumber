#!/usr/bin/env python3
"""Export Mario benchmark metadata and scenario-profile coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.data_generator import TASK_NAMES, TASK_THRESHOLDS, benchmark_metadata
from server.pipeline_doctor_environment import PipelineDoctorEnvironment


def collect_initial_score_stats(seeds: list[int]) -> dict[str, object]:
    env = PipelineDoctorEnvironment()
    payload: dict[str, object] = {}
    for split in ("train", "eval"):
        split_rows: dict[str, object] = {}
        for task_id in sorted(TASK_NAMES):
            scores: list[float] = []
            profiles: dict[str, int] = {}
            for seed in seeds:
                observation = env.reset(seed=seed, task_id=task_id, split=split)
                scores.append(float(observation.current_score))
                profiles[observation.scenario_profile] = profiles.get(observation.scenario_profile, 0) + 1
            split_rows[f"task_{task_id}"] = {
                "name": TASK_NAMES[task_id],
                "threshold": TASK_THRESHOLDS[task_id],
                "initial_score_mean": round(mean(scores), 4),
                "initial_score_min": round(min(scores), 4),
                "initial_score_max": round(max(scores), 4),
                "profiles_seen": profiles,
            }
        payload[split] = split_rows
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Export benchmark metadata and profile coverage.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report = {
        "benchmark_metadata": benchmark_metadata(),
        "initial_score_stats": collect_initial_score_stats(args.seeds),
    }
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
