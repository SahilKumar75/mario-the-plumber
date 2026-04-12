from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def _run_command(*args: str) -> str:
    return _run_command_with_env(*args)


def _run_command_with_env(*args: str, include_hf_token: bool = True) -> str:
    env = os.environ.copy()
    if include_hf_token:
        env.setdefault("API_KEY", "test-token")
        env.setdefault("HF_TOKEN", "test-token")
    else:
        env.pop("API_KEY", None)
        env.pop("HF_TOKEN", None)
    completed = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return completed.stdout


def _run_json_command(*args: str) -> dict[str, object]:
    return json.loads(_run_command(*args))


def _run_json_command_with_env(*args: str, include_hf_token: bool = True) -> dict[str, object]:
    return json.loads(_run_command_with_env(*args, include_hf_token=include_hf_token))


def _parse_bracket_protocol_line(line: str) -> tuple[str, dict[str, str]]:
    raw = line.strip()
    if not raw.startswith("[") or "]" not in raw:
        raise ValueError(f"malformed bracket protocol line: {raw}")
    tag, _, tail = raw[1:].partition("]")
    payload: dict[str, str] = {}
    for token in tail.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        payload[key] = value
    return tag, payload


def test_inference_cli_heuristic_eval_smoke() -> None:
    lines = _run_command(
        "-m",
        "inference",
        "--seed",
        "1",
        "--split",
        "eval",
        "--policy-mode",
        "heuristic",
    ).splitlines()
    protocol = [_parse_bracket_protocol_line(line) for line in lines if line.strip()]

    starts = [payload for tag, payload in protocol if tag == "START"]
    assert len(starts) == 3
    assert [payload["task"] for payload in starts] == ["easy", "medium", "hard"]
    assert all("env" in payload for payload in starts)
    assert all("model" in payload for payload in starts)

    blocks: list[tuple[dict[str, str], list[dict[str, str]], dict[str, str]]] = []
    index = 0
    while index < len(protocol):
        tag, start_payload = protocol[index]
        assert tag == "START"
        index += 1
        step_payloads: list[dict[str, str]] = []
        while index < len(protocol) and protocol[index][0] == "STEP":
            step_payloads.append(protocol[index][1])
            index += 1
        assert index < len(protocol)
        end_tag, end_payload = protocol[index]
        assert end_tag == "END"
        index += 1
        blocks.append((start_payload, step_payloads, end_payload))

    assert len(blocks) == 3
    for _, step_payloads, end_payload in blocks:
        assert step_payloads
        step_numbers = [int(step["step"]) for step in step_payloads]
        assert step_numbers == list(range(1, len(step_payloads) + 1))
        assert all("action" in step for step in step_payloads)
        assert all(step["action"] for step in step_payloads)
        assert all("reward" in step for step in step_payloads)
        assert all("done" in step for step in step_payloads)
        assert all("error" in step for step in step_payloads)
        assert int(end_payload["steps"]) == len(step_payloads)
        assert "score" in end_payload
        assert 0.01 <= float(end_payload["score"]) <= 0.99
        rewards = [value for value in end_payload["rewards"].split(",") if value]
        assert len(rewards) == len(step_payloads)
        assert end_payload["success"] in {"true", "false"}


def test_inference_cli_json_fallback_mode() -> None:
    payload = _run_json_command(
        "-m",
        "inference",
        "--seed",
        "1",
        "--split",
        "eval",
        "--policy-mode",
        "heuristic",
        "--stdout-protocol",
        "json",
    )

    assert payload["status"] == "complete"
    assert payload["policy_mode"] == "heuristic"
    assert payload["scenario_split"] == "eval"
    assert len(payload["results"]) == 5


def test_inference_cli_trained_mode_smoke() -> None:
    payload = _run_json_command(
        "-m",
        "inference",
        "--seed",
        "2",
        "--split",
        "eval",
        "--policy-mode",
        "trained",
        "--stdout-protocol",
        "json",
    )

    assert payload["status"] == "complete"
    assert payload["policy_mode"] == "trained"
    assert payload["scenario_split"] == "eval"
    assert len(payload["results"]) == 5
    assert any(key.startswith("trained") for key in payload["action_source_totals"])


def test_inference_cli_hybrid_without_hf_token_falls_back_to_heuristic() -> None:
    payload = _run_json_command_with_env(
        "-m",
        "inference",
        "--seed",
        "1",
        "--split",
        "eval",
        "--policy-mode",
        "hybrid",
        "--stdout-protocol",
        "json",
        include_hf_token=False,
    )

    assert payload["status"] == "complete"
    assert payload["policy_mode"] == "hybrid"
    assert payload["model_name"] is None
    assert int(payload["action_source_totals"].get("heuristic_no_client", 0)) > 0


def test_inference_cli_pure_llm_without_hf_token_fails() -> None:
    env = os.environ.copy()
    env.pop("API_KEY", None)
    env.pop("HF_TOKEN", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "inference",
            "--seed",
            "1",
            "--split",
            "eval",
            "--policy-mode",
            "pure-llm",
            "--stdout-protocol",
            "json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode != 0
    assert "API_KEY (or HF_TOKEN) environment variable is required for pure-llm policy mode" in (
        completed.stderr + completed.stdout
    )


def test_benchmark_models_cli_writes_json_and_csv(tmp_path) -> None:
    json_out = tmp_path / "benchmark_runs.json"
    csv_out = tmp_path / "benchmark_runs.csv"
    payload = _run_json_command(
        "-m",
        "scripts.benchmark_models",
        "--seeds",
        "1",
        "2",
        "--splits",
        "train",
        "eval",
        "--policies",
        "heuristic",
        "--format",
        "json",
        "--json-out",
        str(json_out),
        "--csv-out",
        str(csv_out),
    )

    assert json_out.exists()
    assert csv_out.exists()
    assert payload["runtime"]["benchmark_version"] == "2.1"
    assert payload["rows"]
    assert payload["generalization_gaps"]
    assert json.loads(json_out.read_text(encoding="utf-8"))["rows"] == payload["rows"]


def test_benchmark_models_cli_compares_trained_and_heuristic() -> None:
    payload = _run_json_command(
        "-m",
        "scripts.benchmark_models",
        "--seeds",
        "1",
        "2",
        "--splits",
        "eval",
        "--policies",
        "heuristic",
        "trained",
        "--format",
        "json",
    )

    policies = {row["policy"] for row in payload["rows"]}
    assert {"heuristic", "trained"}.issubset(policies)


def test_benchmark_adaptation_cli_reports_heldout_profiles() -> None:
    payload = _run_json_command(
        "-m",
        "scripts.benchmark_adaptation",
        "--policy-mode",
        "heuristic",
        "--seeds",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
    )

    assert payload["runtime"]["benchmark_version"] == "2.1"
    assert payload["heldout_task3_seeds"]
    assert payload["heldout_task4_seeds"]
    assert payload["heldout_task5_seeds"]
    assert len(payload["heldout_task3_seeds"]) < len(payload["seeds"])
    assert len(payload["heldout_task4_seeds"]) < len(payload["seeds"])
    assert len(payload["heldout_task5_seeds"]) < len(payload["seeds"])
    assert set(payload["heldout_task3_seeds"]).issubset(set(payload["seeds"]))
    assert set(payload["heldout_task4_seeds"]).issubset(set(payload["seeds"]))
    assert set(payload["heldout_task5_seeds"]).issubset(set(payload["seeds"]))
    assert payload["heldout_family_gap_task3"] > 0.03
    assert payload["heldout_family_gap_task4"] > 0.05
    assert payload["heldout_family_gap_task5"] >= 0.0
    assert all("profile_family" in item and "novelty_axes" in item for item in payload["eval_task3_profiles"].values())
    assert all("profile_family" in item and "novelty_axes" in item for item in payload["eval_task4_profiles"].values())
    assert all("profile_family" in item and "novelty_axes" in item for item in payload["eval_task5_profiles"].values())
    assert any(
        str(item["profile_family"]).startswith("heldout_")
        for item in payload["eval_task3_profiles"].values()
    )
    assert any(
        str(item["profile_family"]).startswith("familiar_")
        for item in payload["eval_task3_profiles"].values()
    )
    assert any(
        str(item["profile_family"]).startswith("heldout_")
        for item in payload["eval_task4_profiles"].values()
    )
    assert any(
        str(item["profile_family"]).startswith("familiar_")
        for item in payload["eval_task4_profiles"].values()
    )
    assert any(
        str(item["profile_family"]).startswith("heldout_")
        for item in payload["eval_task5_profiles"].values()
    )
    assert any(
        str(item["profile_family"]).startswith("familiar_")
        for item in payload["eval_task5_profiles"].values()
    )


def test_export_benchmark_metadata_cli_writes_output(tmp_path) -> None:
    output = tmp_path / "benchmark_metadata.json"
    payload = _run_json_command(
        "-m",
        "scripts.export_benchmark_metadata",
        "--seeds",
        "1",
        "2",
        "--output",
        str(output),
    )

    assert output.exists()
    assert payload["runtime"]["benchmark_version"] == "2.1"
    assert "benchmark_metadata" in payload
    assert "initial_score_stats" in payload
