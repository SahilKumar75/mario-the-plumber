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
    return parser.parse_args()


def train_policy(seeds: list[int], split: str) -> dict[str, Any]:
    counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    payload_lookup: dict[tuple[str, str, str], dict[str, object]] = {}

    for seed in seeds:
        for task_id in sorted(TASK_THRESHOLDS):
            env = PipelineDoctorEnvironment()
            observation = env.reset(seed=seed, task_id=task_id, split=split)

            for _ in range(MAX_STEPS[task_id]):
                if env.state.done:
                    break

                action = heuristic_action_for(task_id, observation)
                signature = observation_signature(task_id, observation)
                action_payload = action.model_dump(exclude_none=True)
                action_key = json.dumps(action_payload, sort_keys=True)
                task_key = str(task_id)

                counts[task_key][signature][action_key] += 1
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
        "version": "1.0",
        "algorithm": "behavior_cloning_count_table",
        "training_split": split,
        "seeds": seeds,
        "signature_counts": {
            task_key: len(task_policy)
            for task_key, task_policy in task_policies.items()
        },
        "samples_collected": samples_collected,
        "task_policies": task_policies,
    }


def main() -> None:
    args = parse_args()
    artifact = train_policy(args.seeds, args.split)
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
