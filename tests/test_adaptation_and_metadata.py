from __future__ import annotations

import json

from scripts.benchmark_adaptation import discover_eval_profiles, discover_heldout_seeds
from scripts.export_benchmark_metadata import collect_initial_score_stats
from server.pipeline_doctor_environment import PipelineDoctorEnvironment


def test_heldout_task4_seed_discovery_matches_environment_flag() -> None:
    seeds = [1, 2, 3, 4, 5, 6]
    heldout = discover_heldout_seeds(4, seeds)

    assert heldout
    assert len(heldout) < len(seeds)
    env = PipelineDoctorEnvironment()
    for seed in seeds:
        observation = env.reset(task_id=4, split="eval", seed=seed)
        assert observation.heldout_profile_family is (seed in heldout)
        if observation.heldout_profile_family:
            assert observation.scenario_profile.startswith("heldout_task4_")
        else:
            assert not observation.scenario_profile.startswith("heldout_task4_")


def test_heldout_task3_seed_discovery_matches_environment_flag() -> None:
    seeds = [1, 2, 3, 4, 5, 6]
    heldout = discover_heldout_seeds(3, seeds)

    assert heldout
    assert len(heldout) < len(seeds)
    env = PipelineDoctorEnvironment()
    for seed in seeds:
        observation = env.reset(task_id=3, split="eval", seed=seed)
        assert observation.heldout_profile_family is (seed in heldout)
        if observation.heldout_profile_family:
            assert observation.scenario_profile.startswith("heldout_task3_")
        else:
            assert not observation.scenario_profile.startswith("heldout_task3_")


def test_heldout_task5_seed_discovery_matches_environment_flag() -> None:
    seeds = [1, 2, 3, 4, 5, 6]
    heldout = discover_heldout_seeds(5, seeds)

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


def test_task4_adaptation_profile_catalog_exposes_family_and_novelty_axes() -> None:
    profiles = discover_eval_profiles(4, [1, 2, 3, 4, 5, 6])

    assert any(item["profile_family"] == "heldout_incremental" for item in profiles.values())
    assert any(item["profile_family"] == "familiar_incremental" for item in profiles.values())
    assert all(item["novelty_axes"] for item in profiles.values())
    assert any(str(item["profile"]).startswith("heldout_task4_") for item in profiles.values())


def test_task3_adaptation_profile_catalog_exposes_family_and_novelty_axes() -> None:
    profiles = discover_eval_profiles(3, [1, 2, 3, 4, 5, 6])

    assert any(item["profile_family"] == "heldout_referential" for item in profiles.values())
    assert any(item["profile_family"] == "familiar_referential" for item in profiles.values())
    assert all(item["novelty_axes"] for item in profiles.values())
    assert any(str(item["profile"]).startswith("heldout_task3_") for item in profiles.values())


def test_task5_adaptation_profile_catalog_exposes_family_and_novelty_axes() -> None:
    profiles = discover_eval_profiles(5, [1, 2, 3, 4, 5, 6])

    assert any(item["profile_family"] == "heldout_temporal" for item in profiles.values())
    assert any(item["profile_family"] == "familiar_temporal" for item in profiles.values())
    assert all(item["novelty_axes"] for item in profiles.values())
    assert any(str(item["profile"]).startswith("heldout_temporal_") for item in profiles.values())
