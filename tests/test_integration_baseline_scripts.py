from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from benchmark.inference_protocol import PROTOCOL_VERSION, parse_protocol_lines


ROOT = Path(__file__).resolve().parents[1]


def _run_command(*args: str) -> str:
    completed = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _run_json_command(*args: str) -> dict[str, object]:
    return json.loads(_run_command(*args))


def test_inference_cli_heuristic_eval_smoke() -> None:
    transcript = parse_protocol_lines(
        _run_command(
            "-m",
            "inference",
            "--seed",
            "1",
            "--split",
            "eval",
            "--policy-mode",
            "heuristic",
        ).splitlines()
    )
    payload = transcript["end"]

    assert transcript["start"]["protocol_version"] == PROTOCOL_VERSION
    assert transcript["start"]["run_id"]
    assert len(transcript["steps"]) == 5
    assert sorted({int(step["task_id"]) for step in transcript["steps"]}) == [1, 2, 3, 4, 5]
    assert all(step.get("event") == "task_complete" for step in transcript["steps"])
    assert payload["protocol_version"] == PROTOCOL_VERSION
    assert payload["run_id"] == transcript["start"]["run_id"]
    assert payload["status"] == "complete"
    assert payload["policy_mode"] == "heuristic"
    assert payload["scenario_split"] == "eval"
    assert len(payload["results"]) == 5
    assert all("scenario_profile" in result for result in payload["results"])


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
    assert payload["heldout_family_gap_task3"] > 0.1
    assert payload["heldout_family_gap_task4"] > 0.05
    assert payload["heldout_family_gap_task5"] > 0.4
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
