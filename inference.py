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

from openai import OpenAI

from benchmark.catalog import MAX_STEPS, TASK_THRESHOLDS
from benchmark.inference_protocol import PROTOCOL_VERSION
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


def _protocol_model_label(policy_mode: str, model_name_override: str | None) -> str:
    if model_name_override:
        return model_name_override
    if policy_mode == "pure-llm":
        return (os.getenv("MODEL_NAME", MODEL_NAME) or MODEL_NAME).strip() or "unknown"
    return policy_mode


def _emit_bracket_start(*, policy_mode: str, model_name_override: str | None) -> None:
    model_label = _protocol_model_label(policy_mode, model_name_override)
    print(
        f"[START] task=mario_the_plumber env=benchmark model={model_label} protocol={PROTOCOL_VERSION}",
        flush=True,
    )


def _emit_bracket_step(*, step: int, reward: float, done: bool, error: str | None) -> None:
    error_value = "null" if not error else str(error)
    print(
        f"[STEP] step={step} reward={reward:.2f} done={str(done).lower()} error={error_value}",
        flush=True,
    )


def _emit_bracket_end(*, success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_blob = ",".join(f"{value:.2f}" for value in rewards) if rewards else "0.00"
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_blob}",
        flush=True,
    )


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
            "stdout format: strict emits [START]/[STEP]/[END] key=value lines for challenge parsers; "
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
    protocol_rewards: list[float] = []
    protocol_step_count = 0

    def record_protocol_step(event: dict[str, object]) -> None:
        nonlocal protocol_step_count
        if event.get("event") != "task_complete":
            return
        protocol_step_count += 1
        reward = float(event.get("score", 0.0))
        protocol_rewards.append(reward)
        _emit_bracket_step(step=protocol_step_count, reward=reward, done=False, error=None)

    if strict_protocol:
        _emit_bracket_start(policy_mode=args.policy_mode, model_name_override=args.model_name)

    if args.seeds:
        runs: list[dict[str, object]] = []
        for seed in args.seeds:
            progress_callback = None
            if strict_protocol:
                def progress_callback(event: dict[str, object]) -> None:
                    record_protocol_step(event)
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
                record_protocol_step(event)
        payload = run_baseline(
            seed=args.seed,
            split=args.split,
            policy_mode=args.policy_mode,
            model_name=args.model_name,
            progress_callback=progress_callback,
        )

    if strict_protocol:
        if not protocol_rewards:
            if isinstance(payload.get("results"), list):
                protocol_rewards = [float(item.get("score", 0.0)) for item in payload["results"]]
            elif isinstance(payload.get("runs"), list):
                for run in payload["runs"]:
                    protocol_rewards.extend(float(item.get("score", 0.0)) for item in run.get("results", []))

        if "average_score" in payload:
            protocol_score = float(payload.get("average_score", 0.0))
        elif protocol_rewards:
            protocol_score = float(sum(protocol_rewards) / len(protocol_rewards))
        else:
            protocol_score = 0.0

        if isinstance(payload.get("results"), list):
            protocol_success = all(bool(item.get("success")) for item in payload["results"])
        elif isinstance(payload.get("runs"), list):
            protocol_success = all(
                all(bool(item.get("success")) for item in run.get("results", []))
                for run in payload["runs"]
            )
        else:
            protocol_success = payload.get("status") in {"complete", "benchmark"}

        _emit_bracket_end(
            success=bool(protocol_success),
            steps=len(protocol_rewards),
            score=protocol_score,
            rewards=protocol_rewards,
        )
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
