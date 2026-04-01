# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenAI-client baseline and benchmark harness for Mario the Plumber."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import time

from openai import OpenAI

try:
    from .benchmark.catalog import MAX_STEPS, TASK_THRESHOLDS
    from .benchmark.policies import choose_action, next_table, table_should_advance
    from .models import PipelineDoctorAction
    from .server.pipeline_doctor_environment import PipelineDoctorEnvironment
except ImportError:
    from benchmark.catalog import MAX_STEPS, TASK_THRESHOLDS
    from benchmark.policies import choose_action, next_table, table_should_advance
    from models import PipelineDoctorAction
    from server.pipeline_doctor_environment import PipelineDoctorEnvironment

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")


def run_baseline(
    seed: int = 42,
    *,
    split: str = "train",
    policy_mode: str = "hybrid",
    model_name: str | None = None,
) -> dict[str, object]:
    """Run the benchmark baseline over the official Mario tasks."""

    started = time.perf_counter()
    client = _build_client()
    selected_model = model_name or MODEL_NAME
    if policy_mode == "pure-llm" and (client is None or not selected_model):
        raise RuntimeError("pure-llm mode requires MODEL_NAME and HF_TOKEN/API_KEY")

    results: list[dict[str, object]] = []
    action_sources: Counter[str] = Counter()

    for task_id in sorted(TASK_THRESHOLDS):
        env = PipelineDoctorEnvironment()
        observation = env.reset(seed=seed, task_id=task_id, split=split)
        task_sources: Counter[str] = Counter()

        for _ in range(MAX_STEPS[task_id]):
            if env.state.done:
                break

            action, action_source = choose_action(
                client,
                selected_model,
                policy_mode,
                task_id,
                env.state.step_count + 1,
                observation,
            )
            action_sources[action_source] += 1
            task_sources[action_source] += 1
            observation = env.step(action)

            if env.state.done:
                break

            if task_id in (3, 4, 5) and table_should_advance(task_id, env, observation):
                next_stage = next_table(env.state.active_table, task_id=task_id)
                if next_stage:
                    observation = env.step(
                        PipelineDoctorAction(action_id=0, target_column=next_stage)
                    )
                    action_sources["auto_table_switch"] += 1
                    task_sources["auto_table_switch"] += 1

        if not env.state.done:
            observation = env.step(PipelineDoctorAction(action_id=15))
            action_sources["forced_commit"] += 1
            task_sources["forced_commit"] += 1

        results.append(
            {
                "task_id": task_id,
                "score": round(observation.current_score, 4),
                "steps": env.state.step_count,
                "success": bool(env.state.success),
                "scenario_profile": observation.scenario_profile,
                "heldout_profile_family": bool(observation.heldout_profile_family),
                "action_sources": dict(task_sources),
            }
        )

    average_score = round(
        sum(float(result["score"]) for result in results) / max(len(results), 1), 4
    )
    return {
        "status": "complete",
        "policy_mode": policy_mode,
        "scenario_split": split,
        "model_name": selected_model if client is not None else None,
        "results": results,
        "average_score": average_score,
        "action_source_totals": dict(action_sources),
        "runtime_seconds": round(time.perf_counter() - started, 2),
    }


def _build_client() -> OpenAI | None:
    if not API_KEY or not MODEL_NAME:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Mario the Plumber baseline.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for one benchmark run.")
    parser.add_argument(
        "--split",
        choices=["train", "eval"],
        default="train",
        help="Scenario distribution split to evaluate.",
    )
    parser.add_argument(
        "--policy-mode",
        choices=["heuristic", "hybrid", "pure-llm"],
        default="hybrid",
        help="Baseline decision policy.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional model override for LLM-backed modes.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Optional list of seeds to benchmark in sequence.",
    )
    args = parser.parse_args()

    if args.seeds:
        payload = {
            "status": "benchmark",
            "runs": [
                {
                    "seed": seed,
                    **run_baseline(
                        seed=seed,
                        split=args.split,
                        policy_mode=args.policy_mode,
                        model_name=args.model_name,
                    ),
                }
                for seed in args.seeds
            ],
        }
    else:
        payload = run_baseline(
            seed=args.seed,
            split=args.split,
            policy_mode=args.policy_mode,
            model_name=args.model_name,
        )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
