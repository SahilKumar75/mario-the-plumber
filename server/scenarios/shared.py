from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

try:
    from ...benchmark.catalog import (
        FORMAL_TASK_SPECS,
        SYNTHETIC_DATA_NOTES,
        TASK_OBJECTIVE_WEIGHTS,
        patterns_for_profile,
        sample_profile,
    )
except ImportError:
    from benchmark.catalog import (
        FORMAL_TASK_SPECS,
        SYNTHETIC_DATA_NOTES,
        TASK_OBJECTIVE_WEIGHTS,
        patterns_for_profile,
        sample_profile,
    )

__all__ = [
    "FORMAL_TASK_SPECS",
    "SYNTHETIC_DATA_NOTES",
    "TASK_OBJECTIVE_WEIGHTS",
    "Scenario",
    "expected_types",
    "patterns_for_profile",
    "sample_profile",
]


@dataclass(slots=True)
class Scenario:
    task_id: int
    seed: int | None
    broken_tables: dict[str, pd.DataFrame]
    ground_truth_tables: dict[str, pd.DataFrame]
    expected_types: dict[str, dict[str, str]]
    active_table: str
    split: str = "train"
    metadata: dict[str, object] = field(default_factory=dict)


def expected_types(frame: pd.DataFrame) -> dict[str, str]:
    return {column: str(dtype) for column, dtype in frame.dtypes.items()}
