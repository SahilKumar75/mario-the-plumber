from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import pandas as pd


@dataclass(slots=True)
class IncidentFixture:
    broken_tables: dict[str, pd.DataFrame]
    ground_truth_tables: dict[str, pd.DataFrame]
    metadata: dict[str, object] = field(default_factory=dict)


def copy_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {name: frame.copy(deep=True) for name, frame in tables.items()}


def _copy_metadata_value(value: object) -> object:
    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)
    if isinstance(value, dict):
        return {key: _copy_metadata_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_metadata_value(item) for item in value]
    return deepcopy(value)


def pack_metadata(pack: IncidentFixture) -> dict[str, object]:
    return {key: _copy_metadata_value(value) for key, value in pack.metadata.items()}
