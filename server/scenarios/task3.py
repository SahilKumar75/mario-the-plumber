from __future__ import annotations

import numpy as np

from benchmark.catalog import TASK_OBJECTIVE_WEIGHTS
from server.incidents import load_task3_fixture
from server.incidents.shared import copy_tables, pack_metadata
from .profile_routing import select_eval_profile
from .shared import FORMAL_TASK_SPECS, Scenario, expected_types, patterns_for_profile, sample_profile

FAMILIAR_EVAL_PROFILES = [
    "alias_and_encoding_regression",
    "timezone_currency_consistency_incident",
    "sentinel_reference_breakage",
    "cascading_reference_outage",
]

HELDOUT_EVAL_PROFILES = [
    "heldout_task3_contract_alias_family",
    "heldout_task3_dependency_rollup_family",
]


def _select_task3_profile(
    rng: np.random.Generator,
    seed: int | None,
    split: str,
) -> str:
    if split != "eval" or seed is None:
        return sample_profile(3, split, rng)
    return select_eval_profile(
        seed=seed,
        task_id=3,
        familiar_profiles=FAMILIAR_EVAL_PROFILES,
        heldout_profiles=HELDOUT_EVAL_PROFILES,
    )


def generate_task3(
    rng: np.random.Generator,
    seed: int | None,
    split: str,
) -> Scenario:
    profile = _select_task3_profile(rng, seed, split)
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
            "heldout_profile_family": profile.startswith("heldout_task3_"),
            "adaptation_target": "Recover an unseen referential recovery family in one episode.",
            "task_spec": FORMAL_TASK_SPECS[3],
            "tradeoff_weights": TASK_OBJECTIVE_WEIGHTS[3],
        },
    )
