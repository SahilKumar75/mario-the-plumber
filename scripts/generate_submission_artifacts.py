#!/usr/bin/env python3

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"


def run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    run(
        "python3",
        "-m",
        "scripts.benchmark_models",
        "--seeds",
        "1",
        "2",
        "--splits",
        "train",
        "eval",
        "--policies",
        "random",
        "heuristic",
        "trained",
        "--format",
        "json",
        "--json-out",
        str(ASSETS / "benchmark_runs.json"),
        "--csv-out",
        str(ASSETS / "benchmark_runs.csv"),
    )
    run(
        "python3",
        "-m",
        "scripts.export_benchmark_metadata",
        "--seeds",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "--output",
        str(ASSETS / "benchmark_metadata.json"),
    )
    with (ASSETS / "adaptation_report.json").open("w", encoding="utf-8") as handle:
        subprocess.run(
            [
                "python3",
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
            ],
            cwd=ROOT,
            check=True,
            stdout=handle,
        )
    run("python3", "-m", "scripts.generate_visuals")


if __name__ == "__main__":
    main()
