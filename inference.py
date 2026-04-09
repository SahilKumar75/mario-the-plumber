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

from debug_trace import debug_log
from openai import OpenAI

from benchmark.catalog import MAX_STEPS, TASK_NAMES
from benchmark.policies import choose_action, next_table, table_should_advance
from models import PipelineDoctorAction
from server.pipeline_doctor_environment import PipelineDoctorEnvironment


BENCHMARK_TASK_IDS = tuple(sorted(TASK_NAMES))

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
    api_key = (os.getenv("HF_TOKEN") or HF_TOKEN or "").strip()
    selected_model = (model_override or os.getenv("MODEL_NAME", MODEL_NAME) or MODEL_NAME).strip()
    debug_log(
        "resolve_runtime_llm_config",
        api_base_url=api_base_url,
        has_api_key=bool(api_key),
        selected_model=selected_model,
        model_override=model_override,
    )
    return (api_key or None, selected_model or None, api_base_url)


def _validate_runtime_llm_config(api_key: str | None, selected_model: str | None) -> None:
    debug_log(
        "validate_runtime_llm_config",
        has_api_key=bool(api_key),
        selected_model=selected_model,
    )
    if not api_key or not selected_model:
        raise RuntimeError("HF_TOKEN environment variable is required")
    if _invalid_api_key_value(api_key):
        raise RuntimeError(
            "HF_TOKEN contains invalid characters. "
            "Re-export token in a clean shell; hidden Unicode (for example U+2028) often causes this."
        )
    if _contains_non_ascii(selected_model):
        raise RuntimeError("MODEL_NAME contains non-ASCII characters; re-enter MODEL_NAME manually.")


def _policy_requires_llm(policy_mode: str) -> bool:
    return policy_mode in {"hybrid", "pure-llm"}


def _format_action(action: PipelineDoctorAction) -> str:
    payload = action.model_dump(exclude_none=True, exclude_defaults=True)
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _format_error(error: str | None) -> str:
    if not error:
        return "null"
    return "_".join(str(error).split())


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
    debug_log(
        "run_baseline_start",
        seed=seed,
        split=split,
        policy_mode=policy_mode,
        model_name_override=model_name,
    )
    api_key, selected_model, api_base_url = _resolve_runtime_llm_config(model_name)
    client = None
    if _policy_requires_llm(policy_mode):
        _validate_runtime_llm_config(api_key, selected_model)
        client = _build_client(api_key=api_key, selected_model=selected_model, api_base_url=api_base_url)
        if client is None:
            raise RuntimeError("HF_TOKEN environment variable is required")

    results: list[dict[str, object]] = []
    action_sources: Counter[str] = Counter()

    for task_id in BENCHMARK_TASK_IDS:
        debug_log("baseline_task_start", task_id=task_id, split=split, seed=seed)
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
            debug_log(
                "baseline_step",
                task_id=task_id,
                step=env.state.step_count,
                action_source=action_source,
                action=_format_action(action),
                reward=float(observation.reward),
                done=bool(observation.done),
                error=observation.action_result or None,
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "step_complete",
                        "task_id": task_id,
                        "step": env.state.step_count,
                        "action": _format_action(action),
                        "reward": float(observation.reward),
                        "done": bool(observation.done),
                        "error": observation.action_result or None,
                    }
                )

            if env.state.done:
                break

            if task_id in (3, 4, 5) and table_should_advance(task_id, env, observation):
                next_stage = next_table(env.state.active_table, task_id=task_id)
                if next_stage:
                    switch_action = PipelineDoctorAction(action_id=0, target_column=next_stage)
                    observation = env.step(switch_action)
                    action_sources["auto_table_switch"] += 1
                    task_sources["auto_table_switch"] += 1
                    debug_log(
                        "baseline_auto_table_switch",
                        task_id=task_id,
                        step=env.state.step_count,
                        target_column=next_stage,
                        reward=float(observation.reward),
                        done=bool(observation.done),
                        error=observation.action_result or None,
                    )
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "event": "step_complete",
                                "task_id": task_id,
                                "step": env.state.step_count,
                                "action": _format_action(switch_action),
                                "reward": float(observation.reward),
                                "done": bool(observation.done),
                                "error": observation.action_result or None,
                            }
                        )

        if not env.state.done:
            commit_action = PipelineDoctorAction(action_id=15)
            observation = env.step(commit_action)
            action_sources["forced_commit"] += 1
            task_sources["forced_commit"] += 1
            debug_log(
                "baseline_forced_commit",
                task_id=task_id,
                step=env.state.step_count,
                reward=float(observation.reward),
                done=bool(observation.done),
                error=observation.action_result or None,
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "step_complete",
                        "task_id": task_id,
                        "step": env.state.step_count,
                        "action": _format_action(commit_action),
                        "reward": float(observation.reward),
                        "done": bool(observation.done),
                        "error": observation.action_result or None,
                    }
                )

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
        debug_log("baseline_task_complete", **task_result)
        if progress_callback is not None:
            progress_callback({"event": "task_complete", **task_result})

    average_score = round(
        sum(float(result["score"]) for result in results) / max(len(results), 1), 4
    )
    payload = {
        "status": "complete",
        "policy_mode": policy_mode,
        "scenario_split": split,
        "model_name": selected_model if client is not None else None,
        "results": results,
        "average_score": average_score,
        "action_source_totals": dict(action_sources),
        "runtime_seconds": round(time.perf_counter() - started, 2),
    }
    debug_log(
        "run_baseline_complete",
        average_score=payload["average_score"],
        runtime_seconds=payload["runtime_seconds"],
        action_source_totals=payload["action_source_totals"],
    )
    return payload


def _build_client(*, api_key: str | None, selected_model: str | None, api_base_url: str) -> OpenAI | None:
    if not api_key or not selected_model:
        debug_log(
            "build_client_skipped",
            has_api_key=bool(api_key),
            selected_model=selected_model,
            api_base_url=api_base_url,
        )
        return None
    if _invalid_api_key_value(api_key):
        debug_log("build_client_invalid_api_key", api_base_url=api_base_url)
        return None
    if _contains_non_ascii(selected_model):
        debug_log("build_client_invalid_model_name", selected_model=selected_model)
        return None
    debug_log("build_client_success", api_base_url=api_base_url, selected_model=selected_model)
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
        f"[START] task=mario_the_plumber env=benchmark model={model_label}",
        flush=True,
    )


def _emit_bracket_step(
    *,
    task_id: int,
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: str | None,
) -> None:
    error_value = _format_error(error)
    print(
        f"[STEP] task=task_{task_id} step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_value}",
        flush=True,
    )


def _emit_bracket_end(*, success: bool, steps: int, rewards: list[float]) -> None:
    rewards_blob = ",".join(f"{value:.2f}" for value in rewards) if rewards else "0.00"
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_blob}",
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
    payload: dict[str, object] = {}
    protocol_success = False

    def record_protocol_step(event: dict[str, object]) -> None:
        nonlocal protocol_step_count
        if event.get("event") != "step_complete":
            return
        protocol_step_count += 1
        reward = float(event.get("reward", 0.0))
        protocol_rewards.append(reward)
        _emit_bracket_step(
            task_id=int(event.get("task_id", 0) or 0),
            step=protocol_step_count,
            action=str(event.get("action", "null")),
            reward=reward,
            done=bool(event.get("done", False)),
            error=event.get("error"),
        )

    if strict_protocol:
        _emit_bracket_start(policy_mode=args.policy_mode, model_name_override=args.model_name)

    try:
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

        if isinstance(payload.get("results"), list):
            protocol_success = all(bool(item.get("success")) for item in payload["results"])
        elif isinstance(payload.get("runs"), list):
            protocol_success = all(
                all(bool(item.get("success")) for item in run.get("results", []))
                for run in payload["runs"]
            )
        else:
            protocol_success = payload.get("status") in {"complete", "benchmark"}

    finally:
        if strict_protocol:
            _emit_bracket_end(
                success=bool(protocol_success),
                steps=len(protocol_rewards),
                rewards=protocol_rewards,
            )

    if not strict_protocol:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
