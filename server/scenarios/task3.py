from __future__ import annotations

import numpy as np

try:
    from ..incidents import load_task3_fixture
    from ..incidents.shared import copy_tables, pack_metadata
except ImportError:
    from server.incidents import load_task3_fixture
    from server.incidents.shared import copy_tables, pack_metadata

from .shared import Scenario, expected_types, patterns_for_profile, sample_profile


def generate_task3(
    rng: np.random.Generator,
    seed: int | None,
    split: str,
) -> Scenario:
    del rng
    profile = sample_profile(3, split, np.random.default_rng(seed))
    pack = load_task3_fixture(profile, split)
    ground_truth_tables = copy_tables(pack.ground_truth_tables)
    return Scenario(
        task_id=3,
        seed=seed,
        broken_tables=copy_tables(pack.broken_tables),
        ground_truth_tables=ground_truth_tables,
        expected_types={name: expected_types(frame) for name, frame in ground_truth_tables.items()},
        active_table="orders",
        split=split,
        metadata={
            **pack_metadata(pack),
            "scenario_profile": profile,
            "open_world_patterns": patterns_for_profile(profile),
        },
    )
