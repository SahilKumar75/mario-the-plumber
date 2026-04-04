from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _run_json_command(*args: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_inference_cli_heuristic_eval_smoke() -> None:
    payload = _run_json_command("inference.py", "--seed", "1", "--split", "eval", "--policy-mode", "heuristic")

    assert payload["status"] == "complete"
    assert payload["policy_mode"] == "heuristic"
    assert payload["scenario_split"] == "eval"
    assert len(payload["results"]) == 5
    assert all("scenario_profile" in result for result in payload["results"])


def test_benchmark_models_cli_writes_json_and_csv(tmp_path) -> None:
    json_out = tmp_path / "benchmark_runs.json"
    csv_out = tmp_path / "benchmark_runs.csv"
    payload = _run_json_command(
        "scripts/benchmark_models.py",
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


def test_benchmark_adaptation_cli_reports_heldout_profiles() -> None:
    payload = _run_json_command(
        "scripts/benchmark_adaptation.py",
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
    assert payload["heldout_task5_seeds"] == [1, 3, 5]
    assert payload["heldout_family_gap"] > 0.4
    assert all("profile_family" in item and "novelty_axes" in item for item in payload["eval_task5_profiles"].values())


def test_export_benchmark_metadata_cli_writes_output(tmp_path) -> None:
    output = tmp_path / "benchmark_metadata.json"
    payload = _run_json_command(
        "scripts/export_benchmark_metadata.py",
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
