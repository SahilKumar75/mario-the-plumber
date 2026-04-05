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
from typing import Callable
from uuid import uuid4

from openai import OpenAI

from benchmark.catalog import MAX_STEPS, TASK_THRESHOLDS
from benchmark.inference_protocol import PROTOCOL_VERSION, emit_protocol_event
from benchmark.policies import choose_action, next_table, table_should_advance
from models import PipelineDoctorAction
from server.pipeline_doctor_environment import PipelineDoctorEnvironment

DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "nvidia/nemotron-super-49b-v1"
API_BASE_URL = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


def _contains_non_ascii(value: str) -> bool:
    return any(ord(character) > 127 for character in value)


def _contains_whitespace(value: str) -> bool:
    return any(character.isspace() for character in value)


def _invalid_api_key_value(value: str) -> bool:
    return _contains_non_ascii(value) or _contains_whitespace(value)


def _resolve_runtime_llm_config(model_override: str | None) -> tuple[str | None, str | None, str]:
    api_base_url = (os.getenv("API_BASE_URL", API_BASE_URL) or API_BASE_URL).strip()
    api_key = (os.getenv("HF_TOKEN") or HF_TOKEN or os.getenv("API_KEY") or "").strip()
    selected_model = (model_override or os.getenv("MODEL_NAME", MODEL_NAME) or MODEL_NAME).strip()
    return (api_key or None, selected_model or None, api_base_url)


def _validate_pure_llm_config(api_key: str | None, selected_model: str | None) -> None:
    if not api_key or not selected_model:
        raise RuntimeError("pure-llm mode requires MODEL_NAME and HF_TOKEN/API_KEY")
    if _invalid_api_key_value(api_key):
        raise RuntimeError(
            "HF_TOKEN/API_KEY contains invalid characters. "
            "Re-export token in a clean shell; hidden Unicode (for example U+2028) often causes this."
        )
    if _contains_non_ascii(selected_model):
        raise RuntimeError("MODEL_NAME contains non-ASCII characters; re-enter MODEL_NAME manually.")


def run_baseline(
    seed: int = 42,
    *,
    split: str = "train",
    policy_mode: str = "hybrid",
    model_name: str | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Run the benchmark baseline over the official Mario tasks."""

    started = time.perf_counter()
    api_key, selected_model, api_base_url = _resolve_runtime_llm_config(model_name)
    if policy_mode == "pure-llm":
        _validate_pure_llm_config(api_key, selected_model)
    client = _build_client(api_key=api_key, selected_model=selected_model, api_base_url=api_base_url)
    if policy_mode == "pure-llm" and client is None:
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

        task_result = {
            "task_id": task_id,
            "score": round(observation.current_score, 4),
            "steps": env.state.step_count,
            "success": bool(env.state.success),
            "scenario_profile": observation.scenario_profile,
            "heldout_profile_family": bool(observation.heldout_profile_family),
            "action_sources": dict(task_sources),
        }
        results.append(task_result)
        if progress_callback is not None:
            progress_callback({"event": "task_complete", **task_result})

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


def _build_client(*, api_key: str | None, selected_model: str | None, api_base_url: str) -> OpenAI | None:
    if not api_key or not selected_model:
        return None
    if _invalid_api_key_value(api_key):
        return None
    if _contains_non_ascii(selected_model):
        return None
    return OpenAI(base_url=api_base_url, api_key=api_key)


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
        choices=["heuristic", "trained", "hybrid", "pure-llm"],
        default="hybrid",
        help="Baseline decision policy.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional model override for LLM-backed modes.",
    )
    parser.add_argument(
        "--stdout-protocol",
        choices=["strict", "json"],
        default="strict",
        help=(
            "stdout format: strict emits START/STEP/END lines for challenge parsers; "
            "json emits a single JSON payload."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Optional list of seeds to benchmark in sequence.",
    )
    args = parser.parse_args()
    strict_protocol = args.stdout_protocol == "strict"
    protocol_run_id = str(uuid4())

    if strict_protocol:
        emit_protocol_event(
            "START",
            {
                "protocol_version": PROTOCOL_VERSION,
                "run_id": protocol_run_id,
                "policy_mode": args.policy_mode,
                "split": args.split,
                "seed": args.seed,
                "seeds": args.seeds or [],
                "model_name": args.model_name,
            },
        )

    if args.seeds:
        runs: list[dict[str, object]] = []
        for seed in args.seeds:
            progress_callback = None
            if strict_protocol:
                def progress_callback(event: dict[str, object], current_seed: int = seed) -> None:
                    emit_protocol_event("STEP", {"seed": current_seed, **event})
            runs.append(
                {
                    "seed": seed,
                    **run_baseline(
                        seed=seed,
                        split=args.split,
                        policy_mode=args.policy_mode,
                        model_name=args.model_name,
                        progress_callback=progress_callback,
                    ),
                }
            )
        payload = {
            "status": "benchmark",
            "runs": runs,
        }
    else:
        progress_callback = None
        if strict_protocol:
            def progress_callback(event: dict[str, object]) -> None:
                emit_protocol_event("STEP", {"seed": args.seed, **event})
        payload = run_baseline(
            seed=args.seed,
            split=args.split,
            policy_mode=args.policy_mode,
            model_name=args.model_name,
            progress_callback=progress_callback,
        )

    if strict_protocol:
        emit_protocol_event(
            "END",
            {
                "protocol_version": PROTOCOL_VERSION,
                "run_id": protocol_run_id,
                **payload,
            },
        )
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
