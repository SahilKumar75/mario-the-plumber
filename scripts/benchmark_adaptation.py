#!/usr/bin/env python3
"""Benchmark one-shot adaptation to held-out scenario profile families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.runtime import runtime_summary  # noqa: E402
from inference import run_baseline  # noqa: E402
from server.pipeline_doctor_environment import PipelineDoctorEnvironment  # noqa: E402


def discover_heldout_task5_seeds(seeds: list[int]) -> list[int]:
    """Return eval seeds that sample the held-out temporal profile family."""

    heldout: list[int] = []
    for seed in seeds:
        env = PipelineDoctorEnvironment()
        observation = env.reset(seed=seed, task_id=5, split="eval")
        if observation.heldout_profile_family:
            heldout.append(seed)
    return heldout


def discover_task5_eval_profiles(seeds: list[int]) -> dict[int, str]:
    profiles: dict[int, str] = {}
    env = PipelineDoctorEnvironment()
    for seed in seeds:
        observation = env.reset(seed=seed, task_id=5, split="eval")
        profiles[seed] = observation.scenario_profile
    return profiles


def summarize(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(mean(scores), 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark adaptation on held-out profile families.")
    parser.add_argument("--policy-mode", choices=["heuristic", "hybrid", "pure-llm"], default="heuristic")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--model-name", default=None)
    args = parser.parse_args()

    train_scores: list[float] = []
    eval_scores: list[float] = []
    heldout_scores: list[float] = []
    familiar_scores: list[float] = []
    heldout_scores_by_profile: dict[str, list[float]] = {}

    eval_profiles = discover_task5_eval_profiles(args.seeds)
    heldout_seeds = [seed for seed, profile in eval_profiles.items() if profile.startswith("heldout_temporal_")]

    for seed in args.seeds:
        train_run = run_baseline(seed=seed, split="train", policy_mode=args.policy_mode, model_name=args.model_name)
        eval_run = run_baseline(seed=seed, split="eval", policy_mode=args.policy_mode, model_name=args.model_name)
        train_task5 = next(item for item in train_run["results"] if item["task_id"] == 5)
        eval_task5 = next(item for item in eval_run["results"] if item["task_id"] == 5)
        train_scores.append(float(train_task5["score"]))
        eval_scores.append(float(eval_task5["score"]))
        eval_score = float(eval_task5["score"])
        profile = eval_profiles[seed]
        if profile.startswith("heldout_temporal_"):
            heldout_scores.append(eval_score)
            heldout_scores_by_profile.setdefault(profile, []).append(eval_score)
        else:
            familiar_scores.append(eval_score)

    payload: dict[str, Any] = {
        "runtime": runtime_summary(),
        "policy_mode": args.policy_mode,
        "seeds": args.seeds,
        "eval_task5_profiles": eval_profiles,
        "heldout_task5_seeds": heldout_seeds,
        "train_task5": summarize(train_scores),
        "eval_task5": summarize(eval_scores),
        "familiar_eval_task5": summarize(familiar_scores),
        "heldout_profile_family_task5": summarize(heldout_scores),
        "heldout_profile_breakdown_task5": {
            profile: summarize(scores) for profile, scores in sorted(heldout_scores_by_profile.items())
        },
        "adaptation_gap": round(mean(train_scores) - mean(eval_scores), 4) if train_scores and eval_scores else 0.0,
        "heldout_family_gap": round(mean(train_scores) - mean(heldout_scores), 4) if train_scores and heldout_scores else 0.0,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
