"""Run Mario baselines with optional MLflow experiment tracking."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import tempfile

import mlflow

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

INFERENCE = importlib.import_module("inference")
MODEL_NAME = INFERENCE.MODEL_NAME
run_baseline = INFERENCE.run_baseline
try:
    BENCHMARK_VERSION = importlib.import_module("benchmark.catalog").BENCHMARK_VERSION
except ImportError:
    BENCHMARK_VERSION = "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Mario baseline experiments with MLflow tracking."
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for one run.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Optional list of seeds to run as separate MLflow runs.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "eval"],
        default="eval",
        help="Scenario split to evaluate.",
    )
    parser.add_argument(
        "--policy-mode",
        choices=["heuristic", "hybrid", "pure-llm"],
        default="hybrid",
        help="Baseline policy mode to evaluate.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional LLM model override for hybrid or pure-llm modes.",
    )
    parser.add_argument(
        "--tracking-uri",
        default="http://127.0.0.1:5000",
        help="MLflow tracking server URI.",
    )
    parser.add_argument(
        "--experiment",
        default="mario-etl-fixer",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--enable-openai-autolog",
        action="store_true",
        help="Enable MLflow OpenAI autologging for LLM-backed runs.",
    )
    return parser.parse_args()


def configure_mlflow(tracking_uri: str, experiment_name: str, enable_autolog: bool) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    if enable_autolog:
        mlflow.openai.autolog()


def log_run(seed: int, payload: dict[str, object], args: argparse.Namespace) -> None:
    run_name = f"{args.policy_mode}-{args.split}-seed-{seed}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("project", "mario-the-plumber")
        mlflow.set_tag("benchmark_version", BENCHMARK_VERSION)
        mlflow.set_tag("seed", str(seed))
        mlflow.log_param("policy_mode", args.policy_mode)
        mlflow.log_param("scenario_split", args.split)
        mlflow.log_param("seed", seed)
        mlflow.log_param("model_name", args.model_name or MODEL_NAME or "none")
        mlflow.log_param("tracking_mode", "local-dev")
        mlflow.log_metric("average_score", float(payload["average_score"]))
        mlflow.log_metric("runtime_seconds", float(payload["runtime_seconds"]))

        for result in payload["results"]:
            task_id = int(result["task_id"])
            mlflow.log_metric(f"task_{task_id}_score", float(result["score"]))
            mlflow.log_metric(f"task_{task_id}_steps", int(result["steps"]))
            mlflow.log_metric(f"task_{task_id}_success", int(bool(result["success"])))
            mlflow.log_param(
                f"task_{task_id}_profile", str(result.get("scenario_profile", ""))
            )
            mlflow.log_param(
                f"task_{task_id}_heldout",
                str(bool(result.get("heldout_profile_family", False))).lower(),
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "mario_result.json"
            result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(result_path), artifact_path="results")


def main() -> None:
    args = parse_args()
    configure_mlflow(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment,
        enable_autolog=args.enable_openai_autolog,
    )

    seeds = args.seeds or [args.seed]
    outputs: list[dict[str, object]] = []

    for seed in seeds:
        payload = run_baseline(
            seed=seed,
            split=args.split,
            policy_mode=args.policy_mode,
            model_name=args.model_name,
        )
        outputs.append({"seed": seed, **payload})
        log_run(seed=seed, payload=payload, args=args)

    if len(outputs) == 1:
        print(json.dumps(outputs[0], indent=2))
    else:
        print(json.dumps({"status": "tracked-benchmark", "runs": outputs}, indent=2))


if __name__ == "__main__":
    main()
