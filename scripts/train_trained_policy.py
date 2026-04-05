#!/usr/bin/env python3
"""Train a lightweight behavior-cloned baseline policy from heuristic trajectories."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from benchmark.catalog import MAX_STEPS, TASK_THRESHOLDS
from benchmark.policies.heuristics import heuristic_action_for
from benchmark.policies.trained import ARTIFACT_PATH, observation_signature
from models import PipelineDoctorAction
from server.pipeline_doctor_environment import PipelineDoctorEnvironment


DEFAULT_MIN_COUNT_BY_TASK = {
    "1": 1,
    "2": 1,
    "3": 2,
    "4": 2,
    "5": 6,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightweight baseline policy artifact.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(1, 81)),
        help="Seeds to use for training trajectories.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "eval"],
        default="train",
        help="Scenario split used for collecting demonstrations.",
    )
    parser.add_argument(
        "--output",
        default=str(ARTIFACT_PATH),
        help="Output path for the trained policy JSON artifact.",
    )
    parser.add_argument(
        "--task5-oversample",
        type=int,
        default=5,
        help="Vote weight multiplier applied to Task 5 demonstrations.",
    )
    parser.add_argument(
        "--task5-extra-seed-max",
        type=int,
        default=240,
        help="Largest seed to include for Task 5-specific trajectory collection.",
    )
    parser.add_argument(
        "--task5-min-match-count",
        type=int,
        default=6,
        help="Minimum Task 5 vote count required to use a trained action at inference.",
    )
    return parser.parse_args()


def _task_seed_schedule(task_id: int, seeds: list[int], task5_extra_seed_max: int) -> list[int]:
    seed_set = sorted(set(seeds))
    if task_id != 5 or task5_extra_seed_max <= 0:
        return seed_set
    if not seed_set:
        return []
    max_seed = seed_set[-1]
    if task5_extra_seed_max <= max_seed:
        return seed_set
    return sorted(set(seed_set + list(range(max_seed + 1, task5_extra_seed_max + 1))))


def train_policy(
    seeds: list[int],
    split: str,
    *,
    task5_oversample: int,
    task5_extra_seed_max: int,
    task5_min_match_count: int,
) -> dict[str, Any]:
    counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    payload_lookup: dict[tuple[str, str, str], dict[str, object]] = {}
    seed_schedule = {
        str(task_id): _task_seed_schedule(task_id, seeds, task5_extra_seed_max)
        for task_id in sorted(TASK_THRESHOLDS)
    }

    for task_id in sorted(TASK_THRESHOLDS):
        task_key = str(task_id)
        vote_weight = max(task5_oversample, 1) if task_id == 5 else 1
        for seed in seed_schedule[task_key]:
            env = PipelineDoctorEnvironment()
            observation = env.reset(seed=seed, task_id=task_id, split=split)

            for _ in range(MAX_STEPS[task_id]):
                if env.state.done:
                    break

                action = heuristic_action_for(task_id, observation)
                signature = observation_signature(task_id, observation)
                action_payload = action.model_dump(exclude_none=True)
                action_key = json.dumps(action_payload, sort_keys=True)

                counts[task_key][signature][action_key] += vote_weight
                payload_lookup[(task_key, signature, action_key)] = action_payload

                observation = env.step(action)
                if env.state.done:
                    break

            if not env.state.done:
                env.step(PipelineDoctorAction(action_id=15))

    task_policies: dict[str, dict[str, dict[str, object]]] = {}
    samples_collected = 0

    for task_key, signatures in counts.items():
        task_policy: dict[str, dict[str, object]] = {}
        for signature, action_votes in signatures.items():
            best_action_key, best_count = max(
                action_votes.items(),
                key=lambda item: (item[1], item[0]),
            )
            task_policy[signature] = {
                "action": payload_lookup[(task_key, signature, best_action_key)],
                "count": best_count,
            }
            samples_collected += sum(action_votes.values())
        task_policies[task_key] = task_policy

    return {
        "version": "1.1",
        "algorithm": "behavior_cloning_count_table",
        "training_split": split,
        "seeds": seeds,
        "seed_schedule": {
            task_key: [int(seed) for seed in task_seeds]
            for task_key, task_seeds in seed_schedule.items()
        },
        "task5_oversample": max(task5_oversample, 1),
        "task5_extra_seed_max": task5_extra_seed_max,
        "signature_counts": {
            task_key: len(task_policy)
            for task_key, task_policy in task_policies.items()
        },
        "samples_collected": samples_collected,
        "min_count_by_task": {
            **DEFAULT_MIN_COUNT_BY_TASK,
            "5": max(task5_min_match_count, 1),
        },
        "task_policies": task_policies,
    }


def main() -> None:
    args = parse_args()
    artifact = train_policy(
        args.seeds,
        args.split,
        task5_oversample=args.task5_oversample,
        task5_extra_seed_max=args.task5_extra_seed_max,
        task5_min_match_count=args.task5_min_match_count,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "trained",
                "output": str(output),
                "signature_counts": artifact["signature_counts"],
                "samples_collected": artifact["samples_collected"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
