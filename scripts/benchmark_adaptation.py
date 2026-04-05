#!/usr/bin/env python3
"""Benchmark one-shot adaptation to held-out scenario profile families."""

from __future__ import annotations

import argparse
import json
from statistics import mean
from typing import Any

from benchmark.runtime import runtime_summary
from inference import run_baseline
from server.pipeline_doctor_environment import PipelineDoctorEnvironment


def summarize(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(mean(scores), 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
    }


def discover_eval_profiles(task_id: int, seeds: list[int]) -> dict[int, dict[str, object]]:
    profiles: dict[int, dict[str, object]] = {}
    env = PipelineDoctorEnvironment()
    for seed in seeds:
        observation = env.reset(seed=seed, task_id=task_id, split="eval")
        manifest = env._scenario_meta.get("incident_manifest", {})
        profiles[seed] = {
            "profile": observation.scenario_profile,
            "profile_family": manifest.get("profile_family", f"familiar_task{task_id}"),
            "novelty_axes": list(manifest.get("novelty_axes", [])),
        }
    return profiles


def discover_heldout_seeds(task_id: int, seeds: list[int]) -> list[int]:
    heldout: list[int] = []
    env = PipelineDoctorEnvironment()
    for seed in seeds:
        observation = env.reset(seed=seed, task_id=task_id, split="eval")
        if observation.heldout_profile_family:
            heldout.append(seed)
    return heldout


def collect_task_adaptation(task_id: int, seeds: list[int], *, policy_mode: str, model_name: str | None) -> dict[str, Any]:
    eval_profiles = discover_eval_profiles(task_id, seeds)
    heldout_seeds = [seed for seed, payload in eval_profiles.items() if str(payload["profile_family"]).startswith("heldout_")]
    train_scores: list[float] = []
    eval_scores: list[float] = []
    heldout_scores: list[float] = []
    familiar_scores: list[float] = []
    heldout_scores_by_profile: dict[str, list[float]] = {}

    for seed in seeds:
        train_run = run_baseline(seed=seed, split="train", policy_mode=policy_mode, model_name=model_name)
        eval_run = run_baseline(seed=seed, split="eval", policy_mode=policy_mode, model_name=model_name)
        train_task = next(item for item in train_run["results"] if item["task_id"] == task_id)
        eval_task = next(item for item in eval_run["results"] if item["task_id"] == task_id)
        train_scores.append(float(train_task["score"]))
        eval_scores.append(float(eval_task["score"]))
        eval_score = float(eval_task["score"])
        profile = str(eval_profiles[seed]["profile"])
        if seed in heldout_seeds:
            heldout_scores.append(eval_score)
            heldout_scores_by_profile.setdefault(profile, []).append(eval_score)
        else:
            familiar_scores.append(eval_score)

    label = f"task{task_id}"
    return {
        f"eval_{label}_profiles": eval_profiles,
        f"heldout_{label}_seeds": heldout_seeds,
        f"train_{label}": summarize(train_scores),
        f"eval_{label}": summarize(eval_scores),
        f"familiar_eval_{label}": summarize(familiar_scores),
        f"heldout_profile_family_{label}": summarize(heldout_scores),
        f"heldout_profile_breakdown_{label}": {
            profile: summarize(scores) for profile, scores in sorted(heldout_scores_by_profile.items())
        },
        f"adaptation_gap_{label}": round(mean(train_scores) - mean(eval_scores), 4) if train_scores and eval_scores else 0.0,
        f"heldout_family_gap_{label}": round(mean(train_scores) - mean(heldout_scores), 4) if train_scores and heldout_scores else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark adaptation on held-out profile families.")
    parser.add_argument(
        "--policy-mode",
        choices=["heuristic", "trained", "hybrid", "pure-llm"],
        default="heuristic",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--model-name", default=None)
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "runtime": runtime_summary(),
        "policy_mode": args.policy_mode,
        "seeds": args.seeds,
    }
    payload.update(collect_task_adaptation(3, args.seeds, policy_mode=args.policy_mode, model_name=args.model_name))
    payload.update(collect_task_adaptation(4, args.seeds, policy_mode=args.policy_mode, model_name=args.model_name))
    payload.update(collect_task_adaptation(5, args.seeds, policy_mode=args.policy_mode, model_name=args.model_name))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
