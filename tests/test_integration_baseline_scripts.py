from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def _run_command(*args: str) -> str:
    env = os.environ.copy()
    env.setdefault("HF_TOKEN", "test-token")
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

    assert protocol[0][0] == "START"
    assert protocol[0][1]["env"] == "benchmark"
    assert protocol[0][1]["task"] == "mario_the_plumber"

    steps = [payload for tag, payload in protocol if tag == "STEP"]
    assert steps
    assert sum(1 for step in steps if int(step["step"]) == 0) == 5
    action_steps = [step for step in steps if int(step["step"]) > 0]
    assert all("quality" in step for step in steps)
    for task_id in {"task_1", "task_2", "task_3", "task_4", "task_5"}:
        task_steps = [int(step["step"]) for step in action_steps if step["task"] == task_id]
        if task_steps:
            assert task_steps == list(range(1, len(task_steps) + 1))
    zero_steps = [step for step in steps if int(step["step"]) == 0]
    assert all("action" not in step for step in zero_steps)
    assert all("reward" not in step for step in zero_steps)
    assert all("action" in step for step in action_steps)
    assert all("reward" in step for step in action_steps)
    assert all("done" in step for step in action_steps)

    end_payload = next(payload for tag, payload in protocol if tag == "END")
    assert int(end_payload["steps"]) == len(action_steps)
    assert "success" in end_payload
    assert "rewards" in end_payload


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
