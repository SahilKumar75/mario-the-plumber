from __future__ import annotations

import numpy as np

from server.incidents import load_task5_fixture
from server.incidents.shared import copy_tables, pack_metadata
from .profile_routing import select_eval_profile
from .shared import FORMAL_TASK_SPECS, TASK_OBJECTIVE_WEIGHTS, Scenario, expected_types, patterns_for_profile, sample_profile

FAMILIAR_EVAL_PROFILES = [
    "temporal_rollup_backfill_incident",
    "schema_evolution_backfill_recovery",
    "late_correction_backpressure_incident",
]

HELDOUT_EVAL_PROFILES = [
    "heldout_temporal_schema_extension_family",
    "heldout_temporal_rollup_contract_family",
    "heldout_temporal_correction_replay_family",
]


def _select_task5_profile(
    rng: np.random.Generator,
    seed: int | None,
    split: str,
) -> str:
    if split != "eval" or seed is None:
        return sample_profile(5, split, rng)
    return select_eval_profile(
        seed=seed,
        task_id=5,
        familiar_profiles=FAMILIAR_EVAL_PROFILES,
        heldout_profiles=HELDOUT_EVAL_PROFILES,
    )


def generate_task5(
    rng: np.random.Generator,
    seed: int | None,
    split: str,
) -> Scenario:
    profile = _select_task5_profile(rng, seed, split)
    pack = load_task5_fixture(profile, split)
    ground_truth_tables = copy_tables(pack.ground_truth_tables)
    return Scenario(
        task_id=5,
        seed=seed,
        broken_tables=copy_tables(pack.broken_tables),
        ground_truth_tables=ground_truth_tables,
        expected_types={name: expected_types(frame) for name, frame in ground_truth_tables.items()},
        active_table="source_orders",
        split=split,
        metadata={
            **pack_metadata(pack),
            "scenario_profile": profile,
            "open_world_patterns": patterns_for_profile(profile),
            "heldout_profile_family": profile.startswith("heldout_temporal_"),
            "adaptation_target": "Recover an unseen temporal profile family in one episode.",
            "task_spec": FORMAL_TASK_SPECS[5],
            "tradeoff_weights": TASK_OBJECTIVE_WEIGHTS[5],
        },
    )
