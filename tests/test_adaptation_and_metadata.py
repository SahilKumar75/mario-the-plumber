from __future__ import annotations

import json

from scripts.benchmark_adaptation import discover_heldout_task5_seeds, discover_task5_eval_profiles
from scripts.export_benchmark_metadata import collect_initial_score_stats
from server.pipeline_doctor_environment import PipelineDoctorEnvironment


def test_heldout_task5_seed_discovery_matches_environment_flag() -> None:
    seeds = [1, 2, 3, 4, 5, 6]
    heldout = discover_heldout_task5_seeds(seeds)

    assert heldout
    assert len(heldout) < len(seeds)
    env = PipelineDoctorEnvironment()
    for seed in seeds:
        observation = env.reset(task_id=5, split="eval", seed=seed)
        assert observation.heldout_profile_family is (seed in heldout)
        if observation.heldout_profile_family:
            assert observation.scenario_profile.startswith("heldout_temporal_")
        else:
            assert not observation.scenario_profile.startswith("heldout_temporal_")


def test_export_metadata_collects_per_task_initial_scores() -> None:
    payload = collect_initial_score_stats([1, 2])

    assert sorted(payload) == ["eval", "train"]
    assert payload["eval"]["task_5"]["name"] == "Temporal Rollup Recovery"
    assert payload["train"]["task_3"]["initial_score_mean"] >= 0.0
    json.dumps(payload)


def test_task5_adaptation_profile_catalog_exposes_family_and_novelty_axes() -> None:
    profiles = discover_task5_eval_profiles([1, 2, 3])

    assert profiles[1]["profile_family"] == "heldout_temporal"
    assert profiles[2]["profile_family"] == "familiar_temporal"
    assert profiles[1]["novelty_axes"]
    assert profiles[3]["profile"].startswith("heldout_temporal_")
